from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.config import BasicConvConfig


class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        groups: int,
        relu: bool,
        bn: bool,
        bias: bool,
    ):
        super().__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(
                out_planes,
                eps=BasicConvConfig.eps,
                momentum=BasicConvConfig.momentum,
                affine=BasicConvConfig.affine,
            )
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None

    def forward(self, features: Tensor) -> Tensor:
        x = self.conv(features)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, features: Tensor) -> Tensor:
        return features.view(features.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(
        self,
        gate_channels: int,
        reduction_ratio: int,
        pool_types: list[Literal["avg", "max"]] = ["avg", "max"],
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels),
        )
        self.pool_types = pool_types

    def forward(self, features: Tensor):
        channel_att_sum: Tensor | None = None

        for pool_type in self.pool_types:
            if pool_type == "avg":
                pool = F.avg_pool2d(features, (features.size(2), features.size(3)))
            elif pool_type == "max":
                pool = F.max_pool2d(features, (features.size(2), features.size(3)))
            elif pool_type == "lp":
                pool = F.lp_pool2d(features, 2, (features.size(2), features.size(3)))
            else:
                continue

            channel_att_raw = self.mlp(pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        assert channel_att_sum is not None
        scale = (
            torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(features)
        )
        return features * scale


class ChannelPool(nn.Module):
    def forward(self, features: Tensor) -> Tensor:
        return torch.cat(
            (
                torch.max(features, 1)[0].unsqueeze(1),
                torch.mean(features, 1).unsqueeze(1),
            ),
            dim=1,
        )


class SpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(
            in_planes=BasicConvConfig.in_planes,
            out_planes=BasicConvConfig.out_planes,
            kernel_size=BasicConvConfig.kernel_size,
            stride=BasicConvConfig.stride,
            dilation=BasicConvConfig.dilation,
            padding=(BasicConvConfig.kernel_size - 1) // 2,
            groups=BasicConvConfig.groups,
            relu=BasicConvConfig.relu,
            bn=BasicConvConfig.batch_normalization,
            bias=BasicConvConfig.bias,
        )

    def forward(self, features) -> Tensor:
        x_compress: Tensor = self.compress(features)
        x_out: Tensor = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return features * scale


class CBAM(nn.Module):
    def __init__(
        self,
        gate_channels: int,
        reduction_ratio: int,
        no_spatial: bool,
        pool_types: list[Literal["avg", "max"]] = ["avg", "max"],
    ):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(
            gate_channels=gate_channels,
            reduction_ratio=reduction_ratio,
            pool_types=pool_types,
        )
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, features: Tensor) -> Tensor:
        x_out = self.ChannelGate(features)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
