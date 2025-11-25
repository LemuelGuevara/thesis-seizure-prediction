import math

import torch
import torch.nn as nn
from torch import Tensor


class GatedFusion1D(nn.Module):
    def __init__(self, channel_dim: int):
        super(GatedFusion1D, self).__init__()
        self.conv1 = nn.Conv1d(channel_dim * 2, channel_dim, kernel_size=1, padding=0)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv1d(channel_dim, channel_dim, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, feat1: Tensor, feat2: Tensor) -> Tensor:
        # feat1, feat2: [B, C]
        x = torch.cat([feat1, feat2], dim=1)
        x = x.unsqueeze(2)
        x = self.relu(self.conv1(x))
        gate = self.sigmoid(self.conv2(x))

        # Reshape inputs to match
        feat1 = feat1.unsqueeze(2)
        feat2 = feat2.unsqueeze(2)

        fused: Tensor = gate * feat1 + (1 - gate) * feat2
        return fused.squeeze(2)
