import os
import torch

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

