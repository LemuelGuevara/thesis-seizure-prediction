import torch.nn as nn
from torch import Tensor
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from src.config import MultiSeizureModelConfig
from src.model.classification.efficientnet import EfficientNetWithCBAM
from src.model.fusion.attention_pooling import AttentionPooling
from src.model.fusion.gated_fusion import GatedFusion1D


class MultimodalSeizureModel(nn.Module):
    def __init__(
        self,
        feature_dim=MultiSeizureModelConfig.feature_dim,
        num_classes=MultiSeizureModelConfig.num_clsses,
        use_cbam=True,
    ):
        super(MultimodalSeizureModel, self).__init__()
        if use_cbam:
            self.tf_encoder = EfficientNetWithCBAM()
            self.bis_encoder = EfficientNetWithCBAM()
        else:
            tf_base = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            bis_base = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            self.tf_encoder = tf_base.features
            self.bis_encoder = bis_base.features

        self.fusion = GatedFusion1D(channel_dim=feature_dim)
        self.attention_pool = AttentionPooling(
            in_dim=feature_dim, num_classes=num_classes
        )

    def forward(self, tf_input: Tensor, bis_input: Tensor) -> Tensor:
        # Extract features (now [B, 1280, 7, 7])
        tf_feat = self.tf_encoder(tf_input)
        bis_feat = self.bis_encoder(bis_input)

        # Flatten spatial maps to [B, 1280] by averaging over H and W
        tf_feat_flat = tf_feat.mean(dim=[2, 3])  # Global Average Pooling
        bis_feat_flat = bis_feat.mean(dim=[2, 3])

        # Gated fusion
        fused_feat = self.fusion(tf_feat_flat, bis_feat_flat)  # [B, 1280]

        # Reshape back to [B, 1280, 7, 7] for attention pooling
        fused_feat = fused_feat.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 7, 7)

        # Classification
        logits = self.attention_pool(fused_feat)
        return logits
