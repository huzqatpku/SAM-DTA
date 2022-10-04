import os
import random
import itertools
from collections import OrderedDict
import numpy as np

from utils import log_info


def make_config(config_name, content):
    if os.path.exists(f'configs/{config_name}'):
        out_dir = 'output/' + config_name.split('/')[-1].replace('.yaml', '')
        if os.path.exists(f'{out_dir}/scores/epoch_9.txt'):
            log_info(f'{config_name} is Done, skip')
            return False
        else:
            log_info('Duplicated configs', config_name)
            with open(f'configs/{config_name}', 'r') as reader:
                old_content = reader.read()
            if content == old_content:
                log_info('Rerun this config')
            else:
                print('Old Conifg:\n' + old_content)
                print()
                print('New Config:\n' + content)
                raise Exception('233')
    else:
        log_info('Make config', config_name)
        with open(f'configs/{config_name}', 'w') as writer:
            writer.write(content)
    return True


template = '''
base: ['demo.yaml']
model:
    kind: {0}
    kwargs:
        num_tasks: {1}
        num_layers: {2}
        hidden: {3}
        use_fp: {4}
        use_desc: {5}
        fc_hidden: {8}
dataset:
    raw_data_dir_list: {9}
    processed_data_dir_list: {10}
    train_batch_size: {6}
training:
    criterions: {7}
'''

test_template = """
base: ['{0}']
model:
    resume: output/{1}/ckpts/epoch_9.pth
dataset:
    csv_path: ../dataset/smis.csv
"""


def main():
    random.seed(1000)
    os.makedirs('configs', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    num_tasks = 401
    temp_raw_data_dir = '../datasets/raw_dataset/protein_{:04}'
    temp_processed_data_dir = '../datasets/processed_dataset/protein_{:04}'

    config_dict = OrderedDict()
    config_dict['kind'] = [
        'SMILES_CONV',
    ]

    config_dict['num_tasks'] = [num_tasks]
    config_dict['num_layers'] = [3, ]
    config_dict['hidden'] = [1024, ]
    config_dict['use_fp'] = [True, ]
    config_dict['use_desc'] = [False]
    config_dict['train_batch_size'] = [10, ]
    config_dict['criterion'] = ['MSELoss', ]
    config_dict['fc_hidden'] = [3072]
    config_items = list(itertools.product(*list(config_dict.values())))
    #
    config_names = []
    for item_ind, item in enumerate(config_items):
        cur_config = dict(zip(config_dict.keys(), item))
        if cur_config['num_tasks'] == 1:
            cur_config['criterion'] = [{'kind': cur_config['criterion'], 'kwargs': {}}]
            for task_ind in range(num_tasks):
                raw_data_dir_list = [temp_raw_data_dir.format(task_ind)]
                processed_data_dir_list = [temp_processed_data_dir.format(task_ind)]
                content = template.format(*cur_config.values(), raw_data_dir_list, processed_data_dir_list)
                config_name = f'SMILES_Individual_FP_{task_ind}_' + '_'.join(np.array(item).astype(str)) + '.yaml'
                test_content = test_template.format(config_name, config_name.split('.')[0])
                test_config_name = config_name.split('.')[0] + '_test.yaml'
                if make_config(config_name, content) and make_config(test_config_name, test_content):
                    config_names.append(config_name)
        else:
            cur_config['criterion'] = [{'kind': cur_config['criterion'], 'kwargs': {}} for _ in range(cur_config['num_tasks'])]
            raw_data_dir_list = [temp_raw_data_dir.format(task_ind) for task_ind in range(cur_config['num_tasks'])]
            processed_data_dir_list = [temp_processed_data_dir.format(task_ind) for task_ind in range(cur_config['num_tasks'])]
            content = template.format(*cur_config.values(), raw_data_dir_list, processed_data_dir_list)
            config_name = f'MultiTask_' + '_'.join(np.array(item).astype(str)) + '.yaml'
            test_content = test_template.format(config_name, config_name.split('.')[0])
            test_config_name = config_name.split('.')[0] + '_test.yaml'
            if make_config(config_name, content) and make_config(test_config_name, test_content):
                config_names.append(config_name)
    log_info('Finish')


if __name__ == '__main__':
    main()