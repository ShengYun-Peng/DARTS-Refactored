from typing import NamedTuple
from typing import Tuple
import torch
import torch.nn as nn

from darts.model.backbone import Cell
from darts.model.head import AuxiliaryHeadCIFAR
from darts.genotype import DARTS

class NetworkCIFAR(nn.Module):
    def __init__(
        self,
        C: int,
        num_classes: int,
        layers: int,
        auxiliary: bool,
        genotype: NamedTuple,
        drop_path_prob: float,
    ) -> None:
        super(NetworkCIFAR, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        self._drop_path_prob = drop_path_prob

        stem_multiplier = 3
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr),
        )

        self.cells = nn.ModuleList()
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        reduction_prev = False

        for i in range(layers):
            # reduction: 2 * channels + 0.5 * (H, W)
            if i in [layers//3, 2*layers//3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False

            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)

            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr

            # auxiliary is applied to 2/3 of the depth
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if self._auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        # initial states s0 and s1
        s0 = self.stem(input)
        s1 = self.stem(input)
        logits_aux = None
        for i, cell in enumerate(self.cells):
            cell_output = cell(s0, s1, self._drop_path_prob)
            s0, s1 = s1, cell_output
            if i == 2 * self._layers // 3 and self._auxiliary and self.training:
                logits_aux = self.auxiliary_head(s1)

        # output shape: (N, 576, 8, 8)
        output = self.global_pooling(s1)
        logits = self.classifier(output.squeeze())

        return logits, logits_aux


if __name__ == "__main__":
    x = torch.rand((5, 3, 32, 32)).cuda()
    model = NetworkCIFAR(C=16, num_classes=10, layers=20, auxiliary=True, genotype=DARTS, drop_path_prob=0.2)
    model = model.train().cuda()

    output, output_aux = model(x)
    print(output.shape, output_aux.shape)

