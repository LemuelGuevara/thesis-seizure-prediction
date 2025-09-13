import torch
import torch.nn as nn
import math

class GatedFusion1D(nn.Module):
    def __init__(self, channel_dim):
        super(GatedFusion1D, self).__init__()
        self.conv1 = nn.Conv1d(channel_dim * 2, channel_dim, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv1d(channel_dim, channel_dim, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, feat1, feat2):
        # feat1, feat2: [B, C]
        x = torch.cat([feat1, feat2], dim=1)
        x = x.unsqueeze(2) 
        x = self.relu(self.conv1(x))
        gate = self.sigmoid(self.conv2(x)) 

        # Reshape inputs to match
        feat1 = feat1.unsqueeze(2) 
        feat2 = feat2.unsqueeze(2)

        fused = gate * feat1 + (1 - gate) * feat2
        return fused.squeeze(2)


def main():
    batch_size = 4
    feature_dim = 1280

    #Dummy Values to Test Fusion
    tf_feat = torch.randn(batch_size, feature_dim)
    bispec_feat = torch.randn(batch_size, feature_dim)

    fusion_module = GatedFusion1D(channel_dim=feature_dim)
    fused_output = fusion_module(tf_feat, bispec_feat)

    #[4, 1280] Expected
    print("Input shape:", tf_feat.shape)
    print("Fused output shape:", fused_output.shape)

if __name__ == "__main__":
    main()