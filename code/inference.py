import argparse
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import log_info
from dataset import MyDataset
from multi_task_main import get_configs
import models


def run_test(configs, data_path):
    # Prepare model
    data_name = data_path.split('/')[-1]
    data_id = data_name.split('.')[0]
    model = models.__dict__[configs.model.kind](**configs.model.kwargs)
    model = model.to(configs.device)
    log_info('Use model', model)
    if configs.model.get('resume'):
        log_info('Resuming from', configs.model.resume)
        model.load_state_dict(torch.load(configs.model.resume))
    model.eval()

    dataset = MyDataset(data_path)
    dataloader = DataLoader(
        dataset,
        batch_size=configs.dataset.eval_batch_size, shuffle=False,
        num_workers=0, pin_memory=False, drop_last=False,
    )
    print('dataloader success')
    #
    device = configs.device
    feats = []
    for smiles_list, y_true in tqdm(dataloader):
        with torch.no_grad():
            feat = model(smiles_list, extract_feature=False)
        y_pred_all = None
        for i, k in enumerate(configs.model.kwargs.task_kind):
            y_pred = feat[i].cpu().numpy().astype(np.float32).reshape(-1)
            if y_pred_all is None:
                y_pred_all = y_pred
            else:
                y_pred_all = np.vstack((y_pred_all, y_pred))
        feats.extend(y_pred_all.transpose())
    feats_df = pd.DataFrame(feats)
    feats_df.to_csv(f'./testresult.csv')


def main_test():
    parser = argparse.ArgumentParser(description='zcb_gnn')
    parser.add_argument('--data_path', required=True, type=str)
    parser.add_argument('--config_file', required=True, type=str)
    args = parser.parse_args()
    configs, is_test, _ = get_configs(parser)
    assert is_test
    run_test(configs, args.data_path)


if __name__ == '__main__':
    main_test()
