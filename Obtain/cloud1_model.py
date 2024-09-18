import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn as nn
import numpy as np
import timm
from torchvision.models.feature_extraction import create_feature_extractor

class CustomResNet18(nn.Module):
    def __init__(self):
        super(CustomResNet18, self).__init__()
        self.model = timm.create_model('regnety_002', pretrained=True)
        self.features = nn.Sequential(*list(self.model.children())[:-1])

        # Define the final layers for output
        self.lin = nn.Sequential(
            nn.Linear(self.model.head.fc.in_features, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 2)  # Assuming 2 outputs for steering and throttle
        )

    def forward(self, x):
        # Forward pass through the model
        x = self.model.stem(x)
        x = self.model.s1(x)
        x = self.model.s2(x)
        x = self.model.s3(x)
        x = self.model.s4(x)
        x = self.model.final_conv(x)
        x = self.model.head.global_pool(x)
        x = self.lin(x)
        return x
