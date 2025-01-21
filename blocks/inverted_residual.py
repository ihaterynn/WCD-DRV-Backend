import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------- Inverted Residual ---------------------
class InvertedResidual(nn.Module):
    """
    An inverted residual block adapted from MobileNetV2.
    """
    def __init__(self, in_ch, out_ch, stride=1, expand_ratio=1):
        super(InvertedResidual, self).__init__()
        hidden_dim = int(in_ch * expand_ratio)
        self.use_res_connect = (stride == 1 and in_ch == out_ch)

        layers = []
        if expand_ratio != 1:
            # Pointwise convolution
            layers.append(nn.Conv2d(in_ch, hidden_dim, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))

        # Depthwise convolution
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3,
                                stride=stride, padding=1,
                                groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))

        # Pointwise-linear convolution
        layers.append(nn.Conv2d(hidden_dim, out_ch, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_ch))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
