from typing import NamedTuple
from typing import List
from typing import Tuple

import torch
import torch.nn as nn

from darts.model.operations import *
from darts.utils import drop_path
from darts.genotype import DARTS


class Cell(nn.Module):
    """
    Cell is a basic unit in DARTS.
    There are 7 nodes in each cell for image classification task
    """
    def __init__(
        self,
        genotype: NamedTuple,
        C_prev_prev: int,
        C_prev: int,
        C: int,
        reduction: bool,
        reduction_prev: bool,
    ) -> None :
        """
        Args:
            genotype: cell connections defined in genotypes.py
            C_prev_prev: num of output channels in the two cells before
            C_prev: num of output channels in the last cell
            C: num of channels in the current cell, which is used over all the nodes in the cell
            reduction: whether current cell is a size reduction or not
            reduction_prev: whether previous cell is a size reduction or not
        """
        super(Cell, self).__init__()

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_in=C_prev_prev, C_out=C)
        else:
            self.preprocess0 = ReLUConvBN(C_in=C_prev_prev, C_out=C, kernel_size=1, stride=1, padding=0)
        self.preprocess1 = ReLUConvBN(C_in=C_prev, C_out=C, kernel_size=1, stride=1, padding=0)

        # build the op names and indices based on whether we need reduction or not
        # indices are the node nums used to build the next node
        # each node only has 2 inputs (k = 2) according to the paper
        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat

        self._compile(C, op_names, indices, concat, reduction)

    def _compile(
        self,
        C: int,
        op_names: Tuple,
        indices: Tuple,
        concat: List,
        reduction: bool,
    ):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2  # 4 each node requires 2 previous nodes
        self._concat = concat
        self.multiplier = len(concat)  # 4
        self._indices = indices

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]

    def forward(
        self,
        s0: torch.tensor,
        s1: torch.tensor,
        drop_prob: float,
    ) -> torch.tensor:
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        # initial nodes - future nodes will be computed from 2 inputs and added in sequence
        states = [s0, s1]
        for i in range(self._steps):
            input1 = 2 * i
            input2 = 2 * i  +1
            h1 = states[self._indices[input1]]
            h2 = states[self._indices[input2]]
            op1 = self._ops[input1]
            op2 = self._ops[input2]
            h1 = op1(h1)
            h2 = op2(h2)

            # dropout
            if self.training and drop_prob > 0:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states.append(s)

        # concat all the node outputs
        out = torch.cat([states[i] for i in self._concat], dim=1)
        return out


if __name__ == "__main__":
    # no reduction
    C_prev_prev = C_prev = C_curr = 36
    x0 = torch.rand((5, C_prev_prev, 32, 32)).cuda()
    x1 = torch.rand((5, C_prev, 32, 32)).cuda()

    cell = Cell(DARTS, C_prev_prev, C_prev, C_curr, reduction=False, reduction_prev=False)
    cell = cell.cuda()
    out = cell(x0, x1, drop_prob=0.2)
    print("No reduction: input {} {}, output {}".format(x0.shape, x1.shape, out.shape))

    # reduction curr
    C_curr *= 2
    cell = Cell(DARTS, C_prev_prev, C_prev, C_curr, reduction=True, reduction_prev=False)
    cell = cell.cuda()
    out = cell(x0, x1, drop_prob=0.2)
    print("Reduction: input {} {}, output {}".format(x0.shape, x1.shape, out.shape))

    # reduction prev
    C_prev_prev = 144
    C_prev = 288
    x0 = torch.rand((5, C_prev_prev, 32, 32)).cuda()
    x1 = torch.rand((5, C_prev, 16, 16)).cuda()
    cell = Cell(DARTS, C_prev_prev, C_prev, C_curr, reduction=False, reduction_prev=True)
    cell = cell.cuda()
    out = cell(x0, x1, drop_prob=0.2)
    print("Reduction prev: input {} {}, output {}".format(x0.shape, x1.shape, out.shape))


