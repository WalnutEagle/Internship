import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn as nn
import numpy as np
import timm
from torchvision.models.feature_extraction import create_feature_extractor
from .Communication.Client.client import send_tensor, receive_tensor, run
from .Communication.Server.server import CommunicationService, serve

class CustomResNet18(nn.Module):
    def __init__(self):
        super(CustomResNet18, self).__init__()
        self.model = timm.create_model('regnety_002', pretrained=True)
        self.features = nn.Sequential(*list(self.model.children())[:-1])
        
        self.goal = nn.Sequential(
            nn.Linear(2, int(self.model.head.fc.in_features / 2)),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(int(self.model.head.fc.in_features / 2), int(self.model.head.fc.in_features))
        )
        
        self.lin = nn.Sequential(
            nn.Linear(self.model.head.fc.in_features * 2, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        # Forward pass through the stem and s1 stages only
        x = self.model.stem(x)  # Process input through the stem stage
        x = self.model.s1(x)    # Process input through the s1 stage
        
        # Print the output shape
        print("Output shape from s1:", x.shape)
        
        return x

# Example usage
if __name__ == "__main__":
    # Initialize the model
    model = CustomResNet18()
    
    # Dummy input tensor with shape (batch_size, channels, height, width)
    # Adjust the size according to your input images (e.g., 3 for RGB images, 224x224 pixels)
    dummy_input = torch.randn(1, 3, 224, 224)  # Batch size 1, 3 channels, 224x224 image

    # Forward pass through the model
    output = model(dummy_input)
    
    # Print the final output shape
    print("Final output shape:", output)
