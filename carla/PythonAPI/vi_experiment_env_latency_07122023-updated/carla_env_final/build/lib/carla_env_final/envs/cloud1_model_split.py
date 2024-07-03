import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn as nn
import numpy as np

# Load the pretrained ResNet-18 model


import torch
import torchvision.models as models
import torch.nn as nn


class CustomResNet18(nn.Module):
    def __init__(self):
        super(CustomResNet18, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        
        self.linear =nn.Sequential(
		    nn.Flatten(),
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 2)
        )
        
        
    def forward_mobile(self, x, split):
        # Forward through the first part of the network until layer i
        modules = list(self.resnet18.children()[:-1]) + [self.linear]
        modules = modules[:split] 
        for module in modules:
            x = module(x)
        return x
    
    def forward_cloud(self, x, split):
        # Forward through the second part of the network from layer i
        modules = list(self.resnet18.children()[:-1]) + [self.linear]
        modules = modules[split:] 
        for module in modules:
            x = module(x)
        return x
    
    def forward(self, x):
        # Forward through the entire network
        # Here, you can choose the split point by setting i
        split = 5
        x = self.forward_mobile(x, split)
        x = self.forward_cloud(x, split)
        return x

model = CustomResNet18()

# Load the weights from the file (replace 'path/to/weights_file.pth' with the actual path)
weights_path = r"model_cloud_only.pth"
model = torch.load(weights_path)

#model.load_state_dict(torch.load(weights_path))

# Put the model in evaluation mode
model.eval()

# Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Create a dummy input tensor (replace this with your actual input)
# The input size should be [batch_size, channels, height, width]
input_tensor = torch.randn(1, 3, 576, 512).to(device)

# Perform inference
with torch.no_grad():
    output = model(input_tensor)

print(output)

