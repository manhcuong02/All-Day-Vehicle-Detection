import torch
from torch import nn
from math import ceil
from .blocks import *

base_model = [
    # expand_ratio, channels, repeats, stride, kernel_size
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3],
]

class EfficientNet(nn.Module):
    def __init__(self, in_channels = 3, phi_value = 0, resolution = 224, dropout_rate = 0.2, num_classes = 2): 
        super(EfficientNet, self).__init__()
        
        width_factor, depth_factor = self.calculate_factors(phi_value)
        last_channels = ceil(1280 * width_factor)
        
        self.features = self.create_features(width_factor, depth_factor, last_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_channels, num_classes),
        )  
        
    def create_features(self, width_factor, depth_factor, last_channels):
        channels = int(32 * width_factor)
        features = [CNNBlock(3, channels, 3, stride=2, padding=1)]
        in_channels = channels
        
        for expand_ratio, channels, repeats, stride, kernel_size in base_model:
            out_channels = 4 * ceil(int(channels * width_factor) / 4)
            layers_repeats = ceil(repeats * depth_factor)

            for layer in range(layers_repeats):
                features.append(
                    InvertedResidualBlock(
                        in_channels,
                        out_channels,
                        expand_ratio=expand_ratio,
                        stride=stride if layer == 0 else 1,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,  # if k=1:pad=0, k=3:pad=1, k=5:pad=2
                    )
                )
                in_channels = out_channels

        features.append(
            CNNBlock(in_channels, last_channels, kernel_size=1, stride=1, padding=0)
        )

        return nn.Sequential(*features)
    
    def calculate_factors(self, phi, alpha=1.2, beta=1.1):
        depth_factor = alpha**phi
        width_factor = beta**phi
        return width_factor, depth_factor

    
    def forward(self, x):
        x = self.pool(self.features(x))
        return self.classifier(x.view(x.shape[0], -1))
    
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_examples, num_classes = 4, 10
    x = torch.randn((num_examples, 3, 224, 224)).to(device)
    model = EfficientNet(
        in_channels = 3,
        num_classes = num_classes,
    ).to(device)

    print(model(x).shape) 