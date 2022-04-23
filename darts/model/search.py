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


class Network(nn.Module):
    def __init__(
        self,
        C: int,
        num_classes: int,
        layers: int,
        criterion: nn.Module,
        steps: int=4,
        multiplier: int=4,
        stem_multiplier: int=3,
    ) -> None:
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        reduction_prev = False
        self.cells = nn.ModuleList()
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)

            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()

    def _loss(self, input: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        logits = self(input)
        loss = self._criterion(logits, label)
        return loss

    def _initialize_alphas(self):
        # 2 types of params: non-reduction and reduction alphas
        K = sum(1 for i in range(self._steps) for n in range(2+i))
        N = len(PRIMITIVES)

        self.alphas_normal = 1e-3 * torch.randn(K, N)
        self.alphas_reduce = 1e-3 * torch.randn(K, N)

        self.alphas_normal = torch.tensor(self.alphas_normal, requires_grad=True, device="cuda")
        self.alphas_reduce = torch.tensor(self.alphas_reduce, requires_grad=True, device="cuda")

        self._arch_parameters = list([
            self.alphas_normal,
            self.alphas_reduce
        ])

    def arch_parameters(self) -> List:
        return self._arch_parameters

    def new(self) -> nn.Module:
        """
        copy arch params (alpha) but not the weights
        """
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion)
        model_new = model_new.cuda()
        for new_params, old_params in zip(model_new.arch_parameters, self.arch_parameters):
            new_params.data.copy_(old_params.data)

        return model_new

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        s0 = self.stem(input)
        s1 = self.stem(input)

        for cell in self.cells:
            # ensure sum of alphas in a row = 1
            if cell._reduction:
                weights = F.softmax(self.alphas_reduce, dim=1)
            else:
                weights = F.softmax(self.alphas_normal, dim=1)
            curr_out = cell(s0, s1, weights)
            s0, s1 = s1, curr_out
        out = self.global_pooling(s1)
        logits = self.classifier(out.reshape(out.shape[0], -1))

        return logits

    def genotype(self):
        # TODO
        raise NotImplementedError("genotype not implemented!")


if __name__ == "__main__":
    # no reduction
    C_prev_prev = C_prev = C_curr = 16
    x0 = torch.rand((5, C_prev_prev, 32, 32)).cuda()
    x1 = torch.rand((5, C_prev, 32, 32)).cuda()
    weights = torch.rand((14, 8)).cuda()

    cell = Cell(steps=4, multiplier=4, C_prev_prev=C_prev_prev, C_prev=C_prev, C=C_curr, reduction=False, reduction_prev=False)
    cell = cell.cuda()
    out = cell(x0, x1, weights)
    print("No reduction: input {} {}, output {}".format(x0.shape, x1.shape, out.shape))

    # reduction curr
    C_curr *= 2
    cell = Cell(steps=4, multiplier=4, C_prev_prev=C_prev_prev, C_prev=C_prev, C=C_curr, reduction=True, reduction_prev=False)
    cell = cell.cuda()
    out = cell(x0, x1, weights)
    print("Reduction: input {} {}, output {}".format(x0.shape, x1.shape, out.shape))

    # network
    criterion = nn.CrossEntropyLoss().cuda()
    model = Network(C=16, num_classes=10, layers=8, criterion=criterion).cuda()
    x = torch.rand(5, 3, 32, 32).cuda()
    out = model(x)
    print("Model output shape: ", out.shape)