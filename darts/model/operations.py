import torch
import torch.nn as nn

OPS = {
    "none":         lambda C, stride, affine: Zero(stride),
    "avg_pool_3x3": lambda C, stride, affine: nn.AvgPool2d(kernel_size=3, stride=stride, padding=1, count_include_pad=False),
    "max_pool_3x3": lambda C, stride, affine: nn.MaxPool2d(kernel_size=3, stride=stride, padding=1),
    "sep_conv_3x3": lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    "sep_conv_5x5": lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    "sep_conv_7x7": lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    "dil_conv_3x3": lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    "dil_conv_5x5": lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    "conv_7x1_1x7": lambda C, stride, affine: ReLUConvBN2Conv(C, C, 7, stride, 3, affine),
    "skip_connect": lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
}

class FactorizedReduce(nn.Module):
    """
    Deal with the channel num changes between prev_prev layer and current layer
    It is used as a 0 input node in a cell. The input channels are divided into two groups. 
    Then, they are concatenated together after conv2d
    """
    def __init__(
        self,
        C_in: int,
        C_out: int,
        affine: bool=True,
    ) -> None:
        """
        Args:
            C_in: input channel num, which should be C_prev_prev
            C_out: channel num used in a cell, which should be C_curr
            affine: used in batchnorm
        """
        super(FactorizedReduce, self).__init__()

        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.relu(x)
        group1 = self.conv_1(x)
        group2 = self.conv_2(x[:, :, 1:, 1:])
        out = torch.cat([group1, group2], dim=1)

        return out

class ReLUConvBN(nn.Module):
    """
    ReLU - Convolution - Batchnorm
    """
    def __init__(
        self,
        C_in: int,
        C_out: int,
        kernel_size: int,
        stride: int,
        padding: int,
        affine: bool=True,
    ) -> None:
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.op(x)

class Zero(nn.Module):
    """
    return all zeros
    """
    def __init__(self, stride: int,) -> None:
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x: torch.tensor,) -> torch.tensor:
        return x[:, :, ::self.stride, ::self.stride].mul(0.) if self.stride != 1 else x.mul(0.)

class Identity(nn.Module):
    """
    return itself
    """
    def __init__(self) -> None:
        super(Identity, self).__init__()

    def forward(self, x: torch.tensor) -> torch.tensor:
        return x

class SepConv(nn.Module):
    """
    ReLU - Convolution - Batchnorm twice
    Use group convolution to reduce parameters
    """
    def __init__(
        self,
        C_in: int,
        C_out: int,
        kernel_size: int,
        stride: int,
        padding: int,
        affine: bool=True,
    ) -> None:
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.op(x)

class DilConv(nn.Module):
    """
    add dilation in convolution
    ensure the output (H, W) is the same as input while stride = 1
    """
    def __init__(
        self,
        C_in: int,
        C_out: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        affine: bool=True,
    ) -> None:
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.op(x)

class ReLUConvBN2Conv(nn.Module):
    """
    ReLU - Convolution - Batchnorm with kernel size 7
    """
    def __init__(
        self,
        C_in: int,
        C_out: int,
        kernel_size: int,
        stride: int,
        padding: int,
        affine: bool=True,
    ) -> None:
        super(ReLUConvBN2Conv).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, (1, kernel_size), stride=(1, stride), padding=(0, padding), bias=False),
            nn.Conv2d(C_in, C_out, (kernel_size, 1), stride=(stride, 1), padding=(padding, 0), bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.op(x)


if __name__ == "__main__":
    C_in = 144
    C_out = 72
    x = torch.rand((5, C_in, 16, 16))

    processor = FactorizedReduce(C_in, C_out)
    output = processor(x)
    print(output.shape)





