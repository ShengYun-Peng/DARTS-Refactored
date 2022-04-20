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
