import time
import math
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr

from torch.utils.data import DataLoader
from sklearn.metrics import r2_score  # r2_score(y_true, y_pred), asymmetric
from sklearn.metrics import roc_curve, auc
import scipy
import sklearn.metrics as m


def pearson_r2_score(y_true, y_pred):
    return scipy.stats.pearsonr(y_true, y_pred)[0] ** 2


def log_info(*msg):
    print('[' + time.asctime(time.localtime(time.time())) + ']', *msg)


def sigmoid_func(z):
    z = np.clip(z, -10, 10)
    return 1 / (1 + np.exp(-z))


def pth2npy(dict_path):
    state_dict = torch.load(dict_path, map_location='cpu')
    # print(state_dict['weight'].size())
    res = np.concatenate([state_dict['bias'].cpu().numpy()[:, None], state_dict['weight']], axis=1)
    np.save('{}.npy'.format(dict_path.split('.')[0]), res)
    print("success")


def mse(y_true, y_pred):
    return m.mean_squared_error(y_true, y_pred)


def r2(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0] ** 2


class R2:

    def __init__(self, kind='r2', exp=False, square=False, mean_std=False, mean=0., std=1., hppb_dataset=False):
        self.metric = {
            'r2': r2_score,
            'pearson_r2': pearson_r2_score,
        }[kind]
        assert exp in [True, False]
        self.exp = exp
        assert square in [True, False]
        self.square = square
        assert mean_std in [True, False]
        self.mean_std = mean_std
        self.mean = mean
        self.std = std
        assert hppb_dataset in [True, False]
        self.hppb_dataset = hppb_dataset

    def __call__(self, y_true, y_pred):
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)
        if self.hppb_dataset:
            y_pred = (1 - np.exp(y_pred)) * 100.
        if self.mean_std:
            y_pred = y_pred * self.std + self.mean
        if self.exp:
            y_pred = np.exp(y_pred) - 1
        if self.square:
            y_pred = np.square(y_pred)
        return self.metric(y_true, y_pred)


class R2FromCls:

    def __init__(self, reg_values, kind='r2'):
        self.metric = {
            'r2': r2_score,
            'pearson_r2': pearson_r2_score,
        }[kind]
        # num_classes = len(reg_values)
        self.reg_values = np.array(reg_values)

    def __call__(self, y_true, y_pred):
        y_true = y_true.reshape(-1)
        y_pred = np.argmax(y_pred, axis=1)
        y_pred = self.reg_values[y_pred]
        return self.metric(y_true, y_pred)


class AUC:

    def __init__(self, sigmoid=True):
        assert sigmoid in [True, False]
        self.sigmoid = sigmoid

    def __call__(self, y_true, y_pred):
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)
        if self.sigmoid:
            y_pred = sigmoid_func(y_pred)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        return auc(fpr, tpr)


class BCELoss:

    def __init__(self, sigmoid=True):
        assert sigmoid in [True, False]
        self.sigmoid = sigmoid

    def __call__(self, output, target):
        if self.sigmoid:
            output = output.sigmoid()
        return F.binary_cross_entropy(output, target)


class RegLoss:

    def __init__(self, kind='l1', log=False, sqrt=False, mean_std=False, mean=0., std=1., hppb_dataset=False):
        self.loss_func = {
            'l1': F.l1_loss,
            'l2': F.mse_loss,
        }[kind]
        assert log in [True, False]
        self.log = log
        assert sqrt in [True, False]
        self.sqrt = sqrt
        assert mean_std in [True, False]
        self.mean_std = mean_std
        self.mean = mean
        self.std = std
        assert hppb_dataset in [True, False]
        self.hppb_dataset = hppb_dataset

    def __call__(self, output, target):
        if self.hppb_dataset:
            target = torch.log(1 - target / 100.0)
        if self.mean_std:
            target = (target - self.mean) / self.std
        if self.log:
            target = torch.log(target + 1.)
        if self.sqrt:
            target = torch.sqrt(target)
        return self.loss_func(output, target)


class RegAsClsLoss:

    def __init__(self, kind, value_offsets):
        assert kind in ['ce', 'bce']
        self.kind = kind
        # num_classes = len(value_offsets)
        self.value_offsets = value_offsets

    def __call__(self, output, target):
        '''
        Params:
            output.shape = (bs, num_classes)
            target.shape = (bs, 1)
        '''
        device = target.device
        target = target.cpu().numpy().reshape(-1)
        if self.kind == 'bce':
            target_cls = np.zeros((output.shape[0], output.shape[1]))
        elif self.kind == 'ce':
            target_cls = np.zeros(target.shape[0])
        prev_offset = np.inf
        for i, value_offset in enumerate(self.value_offsets):
            assign_mask = (target >= value_offset) & (target < prev_offset)
            if self.kind == 'bce':
                target_cls[assign_mask, i] = 1.0
                if i >= 1:
                    target_cls[assign_mask, i - 1] = 0.5
                if i < len(self.value_offsets) - 1:
                    target_cls[assign_mask, i + 1] = 0.5
            elif self.kind == 'ce':
                target_cls[assign_mask] = i
            prev_offset = value_offset
        if self.kind == 'bce':
            target_cls = torch.tensor(target_cls).float().to(device)
            return F.binary_cross_entropy_with_logits(output, target_cls)
        elif self.kind == 'ce':
            target_cls = torch.tensor(target_cls).long().to(device)
            return F.cross_entropy(output, target_cls)


class CosineAnnealingWithWarmUp:

    def __init__(self, optimizer, start_lr, base_lr, final_lr, num_iters_per_epoch, num_warmup_epochs,
                 num_cosine_epochs):
        self.optimizer = optimizer
        warmup_lr_schedule = np.linspace(
            start_lr, base_lr,
            num_iters_per_epoch * num_warmup_epochs,
        )
        iters = np.arange(num_iters_per_epoch * num_cosine_epochs)
        cosine_lr_schedule = np.array(
            [
                final_lr + (
                        0.5 * (base_lr - final_lr)
                        * (1 + math.cos(math.pi * t / (num_iters_per_epoch * num_cosine_epochs)))
                )
                for t in iters
            ]
        )
        self.lr_schedule = np.concatenate([warmup_lr_schedule] + [cosine_lr_schedule])
        self.iteration = 0

    def step(self, ):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr_schedule[self.iteration]
        self.iteration += 1


class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch


class InfiniteDataLoader:
    def __init__(self, dataset, weights, batch_size, num_workers):
        super().__init__()

        if weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(weights,
                                                             replacement=True,
                                                             num_samples=batch_size)
        else:
            sampler = torch.utils.data.RandomSampler(dataset,
                                                     replacement=True)

        if weights == None:
            weights = torch.ones(len(dataset))

        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=True)

        self._infinite_iterator = iter(DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        raise ValueError


class FastDataLoader:
    """DataLoader wrapper with slightly improved speed by not respawning worker
    processes at every epoch."""

    def __init__(self, dataset, batch_size, num_workers):
        super().__init__()

        self.dataset = dataset
        self.batch_size = batch_size

        batch_sampler = torch.utils.data.BatchSampler(
            torch.utils.data.RandomSampler(dataset, replacement=False),
            batch_size=batch_size,
            drop_last=False
        )

        self._infinite_iterator = iter(
            DataLoader(
                dataset,
                num_workers=num_workers,
                batch_sampler=_InfiniteSampler(batch_sampler)
            )
        )

        self._length = len(batch_sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self._infinite_iterator)

    def __len__(self):
        return self._length
