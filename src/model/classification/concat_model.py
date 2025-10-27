import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from src.config import MultiSeizureModelConfig
from src.model.classification.efficientnet import EfficientNetWithCBAM
from src.model.fusion.attention_pooling import AttentionPooling


class ConcatModel(nn.Module):
    def __init__(
        self,
        feature_dim=MultiSeizureModelConfig.feature_dim,
        num_classes=MultiSeizureModelConfig.num_clsses,
        use_cbam=True,
    ):
        super(ConcatModel, self).__init__()
        if use_cbam:
            self.tf_encoder = EfficientNetWithCBAM()
            self.bis_encoder = EfficientNetWithCBAM()
        else:
            base_tf = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            base_bis = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            self.tf_encoder = base_tf.features  # returns [B, C, H, W]
            self.bis_encoder = base_bis.features

        self.attention_pool = AttentionPooling(
            in_dim=2 * feature_dim, num_classes=num_classes
        )

    def forward(self, tf_input: Tensor, bis_input: Tensor) -> Tensor:
        tf_feat = self.tf_encoder(tf_input)
        bis_feat = self.bis_encoder(bis_input)

        tf_feat_flat = tf_feat.mean(dim=[2, 3])  # [B, 1280]
        bis_feat_flat = bis_feat.mean(dim=[2, 3])  # [B, 1280]

        fused_feat = torch.cat([tf_feat_flat, bis_feat_flat], dim=1)  # [B, 2560]
        fused_feat = (
            fused_feat.unsqueeze(-1).unsqueeze(-1).repeat(-1, -1, 7, 7)
        )  # [B, 2560, 7, 7]

        logits = self.attention_pool(fused_feat)
        return logits
