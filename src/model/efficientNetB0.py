import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from src.model.cbam import CBAM

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from src.model.cbam import CBAM

class EfficientNetWithCBAM(nn.Module):
    def __init__(self):
        super(EfficientNetWithCBAM, self).__init__()
        self.base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.features = self.base_model.features
        self.cbam = CBAM(gate_channels=1280)

    def forward(self, x):
        x = self.features(x)       # [B, 1280, 7, 7]
        x = self.cbam(x)           # [B, 1280, 7, 7]
        return x                   # ✅ Preserve spatial dimensions

    
def main():
    model = EfficientNetWithCBAM()

    #Dummy Value to Test EfficientNetB0
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)

    #[2, 1280] Expected
    print("Model output shape:", output.shape)  

if __name__ == "__main__":
    main()