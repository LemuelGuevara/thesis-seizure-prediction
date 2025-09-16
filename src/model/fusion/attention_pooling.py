import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AttentionPooling(nn.Module):
    def __init__(self, in_dim, num_classes=2):
        super(AttentionPooling, self).__init__()
        self.attn = nn.Linear(in_dim, 1)
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, features: Tensor) -> Tensor:
        B, C, H, W = features.size()
        x = features.view(B, C, H * W).permute(0, 2, 1)  # [B, N, C] where N = H*W

        # Compute attention weights
        attn_weights = self.attn(x)  # [B, N, 1]

        # Applying softmax
        attn_weights = F.softmax(attn_weights, dim=1)

        # Weighted sum
        x_weighted = (attn_weights * x).sum(dim=1)  # [B, C]

        # Final classification
        logits: Tensor = self.classifier(x_weighted)  # [B, num_classes]
        return logits
