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

class Cell(nn.Module):
    """
    Fully-connected cell: each edge consists of all the operations in PRIMITIVES, as defined in MixedOp
    node 0 inputs: c_{k - 2}, c_{k - 1}
    node 1 inputs: c_{k - 2}, c_{k - 1}, node 0
    node 2 inputs: c_{k - 2}, c_{k - 1}, node 0, node 1
    node 3 inputs: c_{k - 2}, c_{k - 1}, node 0, node 1, node 2
    c_{k}  inputs: node 0, node 1, node 2, node 3
    2 + 3 + 4 + 5 = 14 MixedOp in total
    """
    def __init__(
        self,
        steps: int,  # num of intermediate nodes
        multiplier: int,  # num of nodes to concatenate
        C_prev_prev: int,
        C_prev: int,
        C: int,
        reduction: bool,
        reduction_prev: bool,
    ) -> None:
        super(Cell, self).__init__()
        self._steps = steps
        self._multiplier = multiplier
        self._reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)

        self._ops = nn.ModuleList()
        for i in range(self._steps):
            for j in range(i + 2):
                if reduction and j < 2:
                    stride = 2
                else:
                    stride = 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0: torch.Tensor, s1: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        offset = 0
        states = [s0, s1]
        for _ in range(self._steps):
            s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        out = torch.cat(states[-self._multiplier:], dim=1)

        return out

