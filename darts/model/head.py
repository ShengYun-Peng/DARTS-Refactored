import torch
import torch.nn as nn

class AuxiliaryHeadCIFAR(nn.Module):
    """
    compute the auxiliary loss at 2/3 depth of the total layers
    the input (H, W) is (8, 8)
    """
    def __init__(
        self,
        C: int,
        num_classes: int,
    ) -> None:
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),  # image size = 2 * 2
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.features(x)
        x = self.classifier(x.view(x.size(0),-1))
        return x

if __name__ == "__main__":
    C = 576
    x = torch.rand(5, C, 8, 8)
    model = AuxiliaryHeadCIFAR(C, 10)
    x = x.cuda()
    model = model.cuda()

    out = model(x)
    print(out.shape)

