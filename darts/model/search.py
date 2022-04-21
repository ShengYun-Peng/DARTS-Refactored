import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from darts.model.operations import *
from darts.genotype import PRIMITIVES
from darts.genotype import Genotype

class MixedOp(nn.Module):
    """
    Define all operations performed on one edge
    """
    def __init__(
        self,
        C: int,
        stride: int,
    ) -> None:
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if "pool" in primitive:
                op = nn.Sequential(
                    op,
                    nn.BatchNorm2d(C, affine=False),
                )
            self._ops.append(op)

    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        weighted sum of all operations
        """
        ret = sum(w * op(x) for w, op in zip(weights, self._ops))
        return ret
