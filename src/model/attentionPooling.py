import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPooling(nn.Module):
    def __init__(self, in_dim, num_classes=2):
        super(AttentionPooling, self).__init__()
        self.attn = nn.Linear(in_dim, 1)
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        """
        x: Tensor of shape [B, C, H, W]
        """
        B, C, H, W = x.size()
        x = x.view(B, C, H * W).permute(0, 2, 1)  # [B, N, C] where N = H*W

        # Compute attention weights
        attn_weights = self.attn(x)              # [B, N, 1]
        attn_weights = F.softmax(attn_weights, dim=1)  # Softmax over N

        # Weighted sum
        x_weighted = (attn_weights * x).sum(dim=1)      # [B, C]

        # Final classification
        logits = self.classifier(x_weighted)     # [B, num_classes]
        return logits

def main():
    batch_size = 4
    in_channels = 1280
    height = 7
    width = 7

    dummy_features = torch.randn(batch_size, in_channels, height, width)
    model = AttentionPooling(in_dim=in_channels, num_classes=2)
    logits = model(dummy_features)

    print("Input shape:", dummy_features.shape)
    #[4, 2] Expected
    print("Logits shape:", logits.shape)  

if __name__ == "__main__":
    main()
