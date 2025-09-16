import torch.nn as nn
from torch import Tensor
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from src.config import CBAMConfig

from .cbam import CBAM


class EfficientNetWithCBAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.features = self.base_model.features
        self.cbam = CBAM(
            gate_channels=CBAMConfig.gate_channels,
            reduction_ratio=CBAMConfig.reduction_ratio,
            no_spatial=CBAMConfig.no_spatial,
        )

    def forward(self, features: Tensor) -> Tensor:
        features = self.features(features)  # [B, gate_chanels, 7, 7]
        features = self.cbam(features)  # [B, gate_chanels, 7, 7]
        return features
