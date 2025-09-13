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
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.features(x)
        #Add CBAM to the Model
        x = self.cbam(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # [B, 1280]
        return x
    
def main():
    model = EfficientNetWithCBAM()

    #Dummy Value to Test EfficientNetB0
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)

    #[2, 1280] Expected
    print("Model output shape:", output.shape)  

if __name__ == "__main__":
    main()