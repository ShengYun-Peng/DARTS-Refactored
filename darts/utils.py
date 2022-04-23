import os
from typing import List, Tuple
import torch
import torch.nn as nn
import numpy as np

class AvgrageMeter(object):
    """
    Statistics collector
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val: float, n: int=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def drop_path(x: torch.tensor, drop_prob: float, use_gpu: bool=True) -> torch.tensor:
    """
    Similar to dropout on a portion of samples in a batch
    """
    N = x.shape[0]

    if drop_prob > 0.:
        keep_prob = 1 - drop_prob
        mask = torch.FloatTensor(N, 1, 1, 1).bernoulli_(keep_prob)
        if use_gpu:
            mask = mask.cuda()
        x.div_(keep_prob)
        x.mul_(mask)
    return x

def load(model: nn.Module, model_path: str):
    assert os.path.exists(model_path), model_path
    model.load_state_dict(torch.load(model_path))

def count_parameters_in_MB(model: nn.Module) -> float:
    total_size = 0.
    for name, params in model.named_parameters():
        if "auxiliary" not in name:
            total_size += np.prod(params.shape)
    total_size /= 1e6

    return total_size

def compute_accuracy(output: torch.Tensor, label: torch.Tensor, topk: Tuple=(1,)) -> List:
    N = label.shape[0]
    assert output.shape[0] == N

    maxk = max(topk)
    _, pred = output.topk(k=maxk, dim=1)
    pred = pred.T
    target = label.reshape(1, -1).expand_as(pred)
    assert target.shape == (maxk, N)

    correct = pred.eq(target)

    ret = list()
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum()
        rate = correct_k * 100.0 / N
        ret.append(rate)

    return ret

def save(model: nn.Module, model_path: str):
    torch.save(model.state_dict(), model_path)

