import torch
import torch.nn as nn
from src.model.efficientNetB0 import EfficientNetWithCBAM
from src.model.gated import GatedFusion1D
from src.model.attentionPooling import AttentionPooling

class MultimodalSeizureModel(nn.Module):
    def __init__(self, feature_dim=1280, num_classes=2):
        super(MultimodalSeizureModel, self).__init__()
        self.tf_encoder = EfficientNetWithCBAM()
        self.bis_encoder = EfficientNetWithCBAM()
        self.fusion = GatedFusion1D(channel_dim=feature_dim)
        self.attention_pool = AttentionPooling(in_dim=feature_dim, num_classes=num_classes)

    def forward(self, tf_input, bis_input):
        # Extract features (now [B, 1280, 7, 7])
        tf_feat = self.tf_encoder(tf_input)
        bis_feat = self.bis_encoder(bis_input)

        # Flatten spatial maps to [B, 1280] by averaging over H and W
        tf_feat_flat = tf_feat.mean(dim=[2, 3])      # Global Average Pooling
        bis_feat_flat = bis_feat.mean(dim=[2, 3])

        # Gated fusion
        fused_feat = self.fusion(tf_feat_flat, bis_feat_flat)  # [B, 1280]

        # Reshape back to [B, 1280, 7, 7] for attention pooling
        fused_feat = fused_feat.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 7, 7)

        # Classification
        logits = self.attention_pool(fused_feat)
        return logits


    
def main():
    model = MultimodalSeizureModel()
    tf_input = torch.randn(2, 3, 224, 224)
    bis_input = torch.randn(2, 3, 224, 224)

    logits = model(tf_input, bis_input)
    #[2,2] Expected
    print("Final logits shape:", logits.shape)

if __name__ == "__main__":
    main()