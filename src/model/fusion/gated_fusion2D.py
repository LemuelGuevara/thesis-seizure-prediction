import math
import torch
import torch.nn as nn
from torch import Tensor


class GatedFusion2D(nn.Module):
    def __init__(self, channel_dim: int):
        super(GatedFusion2D, self).__init__()
        self.conv1 = nn.Conv2d(channel_dim * 2, channel_dim, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(channel_dim, channel_dim, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, feat1: Tensor, feat2: Tensor) -> Tensor:
        # feat1, feat2: [B, C, H, W] - e.g., [B, 1280, 7, 7]
        x = torch.cat([feat1, feat2], dim=1)  # [B, 2*C, H, W]
        x = self.relu(self.conv1(x))          # [B, C, H, W]
        gate = self.sigmoid(self.conv2(x))    # [B, C, H, W]

        fused: Tensor = gate * feat1 + (1 - gate) * feat2  # [B, C, H, W]
        return fused