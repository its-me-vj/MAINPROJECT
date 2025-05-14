import torch
import torch.nn as nn
import torchvision.models as models


class CSRNet(nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()
        self.frontend = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:23]
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 1, kernel_size=1)
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return x

    def _initialize_weights(self):
        for m in self.backend.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# Ensure both tuning.py and advanced_test.py use this consistent architecture
model = CSRNet()
print(model)
