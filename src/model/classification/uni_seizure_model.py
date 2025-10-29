import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from src.config import MultiSeizureModelConfig
from src.model.classification.efficientnet import EfficientNetWithCBAM
from src.model.fusion.attention_pooling import AttentionPooling


class UnimodalSeizureModel(nn.Module):
    """
    A unimodal seizure prediction model using EfficientNetB0 (optionally with CBAM)
    followed by spatial attention pooling and a classification layer.
    """

    def __init__(
        self,
        feature_dim=MultiSeizureModelConfig.feature_dim,
        num_classes=MultiSeizureModelConfig.num_clsses,
        use_cbam=True,
    ):
        super(UnimodalSeizureModel, self).__init__()
        # Encoder
        if use_cbam:
            self.encoder = EfficientNetWithCBAM()
        else:
            base = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            self.encoder = base.features

        # Attention-based pooling for classification
        self.attention_pool = AttentionPooling(
            in_dim=feature_dim,
            num_classes=num_classes
        )

    def forward(self, x: Tensor) -> Tensor:
        # Extract feature map: [B, 1280, 7, 7]
        feat = self.encoder(x)

        # Classification through attention pooling
        logits = self.attention_pool(feat)
        return logits
