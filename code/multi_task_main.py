import numpy as np
import os
import torch
import random
import argparse
import json
import yaml
from addict import Dict
import torch.backends.cudnn as cudnn
from utils import log_info, mse, pearson_r2_score
from torch.utils.data import DataLoader
import sklearn.metrics as m
from scipy.stats import pearsonr
from dataset import MyDataset
from utils import log_info
from utils import CosineAnnealingWithWarmUp
from utils import InfiniteDataLoader
import models


def merge_configs(configs, base_configs):
    for key in base_configs:
        if not key in configs:
            configs[key] = base_configs[key]
        elif type(configs[key]) is dict:
            merge_configs(configs[key], base_configs[key])


def build_configs(config_file, loaded_config_files):
    loaded_config_files.append(config_file)
    with open(config_file, 'r') as reader:
        configs = yaml.load(reader, Loader=yaml.Loader)
    for base_config_file in configs['base']:
        base_config_file = os.getcwd() + '/configs/' + base_config_file
        if base_config_file in loaded_config_files:
            continue
        base_configs = build_configs(base_config_file, loaded_config_files)
        merge_configs(configs, base_configs)
    return configs


def clear_configs(configs):
    keys = list(configs.keys())
    for key in keys:
        if type(configs[key]) is dict:
            configs[key] = clear_configs(configs[key])
        elif configs[key] == 'None':
            print('Clear config', key)
            configs.pop(key)
    return configs


def MSE(y_true, y_pred):
    return m.mean_squared_error(y_true, y_pred)


def R2(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0] ** 2


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def run_eval(device, model, dataloaders, use_desc):
    model.eval()
    descriptors = None
    if use_desc:
        descriptors = torch.load('../dataset/descriptors.pt')
    infer_results = 'smiles,y_true,y_pred\n'
    scores = []
    for task_ind, dataloader in enumerate(dataloaders):
        y_true = []
        y_pred = []
        all_smiles_list = []
        for smiles_list, y in dataloader:
            y_true.extend(y.tolist())

            if use_desc:
                descriptors = torch.tensor(np.stack(descriptors[smi] for smi in smiles_list)).to(device)
                assert len(smiles_list) == len(descriptors)
            with torch.no_grad():
                output = model(smiles_list, descriptors)[task_ind]
            y_pred.extend(output.cpu().tolist())
            all_smiles_list.extend(smiles_list)
        y_pred = np.array(y_pred).astype(np.float32).reshape(-1)
        y_true = np.array(y_true).astype(np.float32).reshape(-1)
        score = {
            'MSE': str(round(MSE(y_true, y_pred), 4)),
            'R2': str(round(R2(y_true, y_pred), 4)),
        }
        scores.append(score)
        infer_results += f'Task#{task_ind}\n'
        for smiles, one_y_true, one_y_pred in zip(all_smiles_list, y_true, y_pred):
            infer_results += f'{smiles},{one_y_true},{one_y_pred}\n'
    return scores, infer_results


def run_train(
        tag, epoch, num_epochs, device,
        model, train_iterator, num_iters_per_epoch,
        criterions, optimizer, scheduler,
        log_freq, use_desc,
):
    descriptors = None
    if use_desc:
        descriptors = torch.load('./dataset/descriptors.pt')
    max_num_tasks = 64
    model.train()
    loss_item_list = []
    for step in range(num_iters_per_epoch):
        scheduler.step()
        data_batches = next(train_iterator)
        loss = 0.
        loss_per_task = []
        #
        for task_ind, (smiles_list, y_true) in enumerate(data_batches):
            if (task_ind % max_num_tasks) == (max_num_tasks - 1):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss = 0.
            if use_desc:
                descriptors = torch.tensor(np.stack(descriptors[smi] for smi in smiles_list)).to(device)
                assert len(smiles_list) == len(descriptors)
            output = model(smiles_list, descriptors)[task_ind]
            one_task_loss = criterions[task_ind](output.squeeze(), y_true.cuda())
            loss = loss + one_task_loss
            loss_per_task.append(one_task_loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #
        loss_item_list.append(np.sum(loss_per_task))
        loss_per_task = ','.join([f'{loss_item:.3}' for loss_item in loss_per_task])
        if step % log_freq == (log_freq - 1):
            lr = optimizer.param_groups[0]['lr']
            log_info(
                f'{tag} Train Epoch [{epoch}/{num_epochs}] Iter [{step}/{num_iters_per_epoch}] Loss {loss_per_task} LR {lr:.3e}')
    avg_loss = round(np.mean(loss_item_list), 3)
    if device.type == 'cuda':
        memory = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 3)
    else:
        memory = 0.
    log_info(f'{tag} Train Epoch [{epoch}/{num_epochs}] Mem {memory}MB AvgLoss {avg_loss}')


def get_dataset(raw_data_dir):
    train_set = MyDataset(raw_data_dir + '/train.csv')
    val_set = MyDataset(raw_data_dir + '/val.csv')
    test_set = MyDataset(raw_data_dir + '/test.csv')
    return train_set, val_set, test_set


def train_eval(configs, is_test):
    # Prepare model
    model = models.__dict__[configs.model.kind](**configs.model.kwargs)
    model = model.to(configs.device)
    log_info('Use model', model)
    if configs.model.get('resume'):
        log_info('Resuming from', configs.model.resume)
        model.load_state_dict(torch.load(configs.model.resume))

    train_loaders = []
    eval_train_loaders = []
    val_loaders = []
    test_loaders = []
    raw_data_dir_list = configs.dataset.raw_data_dir_list

    for raw_data_dir in raw_data_dir_list:
        train_set, val_set, test_set = get_dataset(raw_data_dir)
        if not is_test:
            train_loaders.append(
                InfiniteDataLoader(
                    train_set, weights=None,
                    batch_size=configs.dataset.train_batch_size,
                    num_workers=1,  # num_workers of train_loaders should not be too large
                )
            )

        eval_train_loaders.append(
            DataLoader(
                train_set,
                batch_size=configs.dataset.eval_batch_size, shuffle=False,
                num_workers=configs.dataset.num_workers, pin_memory=False, drop_last=False,
                # num_workers of dataloaders should not be too large
            )
        )
        val_loaders.append(
            DataLoader(
                val_set,
                batch_size=configs.dataset.eval_batch_size, shuffle=False,
                num_workers=configs.dataset.num_workers, pin_memory=False, drop_last=False,
                # num_workers of dataloader should not be too large
            )
        )
        test_loaders.append(
            DataLoader(
                test_set,
                batch_size=configs.dataset.eval_batch_size, shuffle=False,
                num_workers=configs.dataset.num_workers, pin_memory=False, drop_last=False,
            )
        )
    train_iterator = zip(*train_loaders)

    optimizer = torch.optim.__dict__[configs.training.optimizer.kind](
        model.parameters(), **configs.training.optimizer.kwargs
    )

    log_info('Use optimizer', optimizer)
    criterions = []
    for criterion_config in configs.training.criterions:
        criterion = {
            'MSELoss': torch.nn.MSELoss,
            'L1Loss': torch.nn.L1Loss,
            'CrossEntropyLoss': torch.nn.CrossEntropyLoss,
        }[criterion_config.kind](**criterion_config.kwargs)
        criterions.append(criterion)
    #
    scheduler = {
        'CosineAnnealingWithWarmUp': CosineAnnealingWithWarmUp,
    }[configs.training.scheduler.kind](
        optimizer, num_iters_per_epoch=configs.training.num_iters_per_epoch,
        **configs.training.scheduler.kwargs
    )
    # is test
    if is_test:
        configs.training.num_epochs = 1
    for epoch in range(configs.training.num_epochs):
        if not is_test:
            run_train(
                configs.tag, epoch, configs.training.num_epochs, configs.device,
                model, train_iterator, configs.training.num_iters_per_epoch,
                criterions, optimizer, scheduler,
                configs.training.log_freq, configs.model.kwargs.use_desc,
            )
        train_scores, train_infer_results = run_eval(configs.device, model, eval_train_loaders,
                                                     configs.model.kwargs.use_desc, )
        val_scores, val_infer_results = run_eval(configs.device, model, val_loaders, configs.model.kwargs.use_desc, )
        test_scores, test_infer_results = run_eval(configs.device, model, test_loaders, configs.model.kwargs.use_desc, )
        log_info(
            f'Epoch {epoch}/{configs.training.num_epochs}\n'
            f'\t\t\t Train {train_scores}\n'
            f'\t\t\t Valid {val_scores}\n'
            f'\t\t\t Test  {test_scores}'
        )
        for phase, infer_results in zip(['train', 'val', 'test'],
                                        [train_infer_results, val_infer_results, test_infer_results]):
            with open(f'{configs.out_dir}/infer_results/epoch_{epoch}_{phase}.txt', 'w') as writer:
                writer.write(infer_results)

        with open(f'{configs.out_dir}/scores/epoch_{epoch}.txt', 'w') as writer:
            writer.write(json.dumps({'train': train_scores, 'val': val_scores, 'test': test_scores}, indent=4))
        torch.save(model.state_dict(), f'{configs.out_dir}/ckpts/epoch_{epoch}.pth')

    return train_scores, val_scores, test_scores


def get_configs(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(description='smiles_conv')
        parser.add_argument('--config_file', required=True, type=str)
    args = parser.parse_args()
    config_file = args.config_file
    is_test = config_file.endswith('test.yaml')

    out_dir = 'output/' + config_file.split('/')[-1].replace('.yaml', '')
    configs = Dict(clear_configs(build_configs(config_file, [])))
    tag = config_file.split('/')[-1].split('.yaml')[0]
    #
    configs.tag = tag
    configs.out_dir = out_dir
    seed_all(configs.seed)
    log_info(f'configs\n{json.dumps(configs, indent=2, ensure_ascii=False)}')
    configs.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_info('device is', configs.device)
    os.makedirs(configs.out_dir + '/infer_results', exist_ok=True)
    os.makedirs(configs.out_dir + '/scores', exist_ok=True)
    os.makedirs(configs.out_dir + '/ckpts', exist_ok=True)
    return configs, is_test


def collect_evaluation(configs):
    for phase in ['train', 'val', 'test']:
        with open(configs.out_dir + f'/infer_results/epoch_9_{phase}.txt',
                  'r') as reader:
            lines = reader.readlines()[1:]
        targets = []
        outputs = []
        for line in lines:
            if line.startswith('Task#'):
                continue
            smi, target, output = line.strip().split(',')
            targets.append(float(target))
            outputs.append(float(output))
        targets = np.array(targets).squeeze()
        outputs = np.array(outputs).squeeze()
        result_mse = mse(targets, outputs)
        result_r2 = pearson_r2_score(targets, outputs)
        with open(configs.out_dir + f'/evaluation.txt', 'a') as writer:
            writer.write(phase + '\t')
            writer.write('mse' + str(result_mse) + '\t')
            writer.write('r2' + str(result_r2) + '\n')


def main():
    configs, is_test = get_configs()
    train_eval(configs, is_test)
    collect_evaluation(configs)


if __name__ == '__main__':
    main()
