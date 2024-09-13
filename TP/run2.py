import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import timm
from torchvision.models.feature_extraction import create_feature_extractor
from PIL import Image
import torchvision.transforms as transforms

# Define the CustomResNet18 class
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

    def forward(self, x, locations):
        x = self.model.stem(x)
        x = self.model.s1(x)
        x = self.model.s2(x)
        x = self.model.s3(x)
        x = self.model.s4(x)
        x = self.model.final_conv(x)
        x = self.model.head.global_pool(x)
        y = self.goal(locations)
        sf = torch.cat((x, y), dim=1)
        sf = self.lin(sf)
        return sf

# Instantiate the model
model = CustomResNet18()

# Load the model weights
model_path = 'path/to/your/model/policy.pth'  # Replace with your actual path
model.load_state_dict(torch.load(model_path))

# Set the model to evaluation mode
model.eval()

# Load the image
image_path = 'path/to/your/image.jpg'  # Replace with your actual image path
image = Image.open(image_path)

# Define the preprocessing transformation
preprocess = transforms.Compose([
    transforms.Resize((1, 4)),  # Resize to match model's expected input size
    transforms.ToTensor(),          # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet mean and std
])

# Preprocess the image
input_image = preprocess(image).unsqueeze(0)  # Add batch dimension

# Example location data (replace with your actual location data)
location_data = torch.tensor([[0.5, -0.2]])  # Shape: [batch_size, 2]
location_data = location_data.float()  # Ensure it's a float tensor

# Move the model and inputs to the appropriate device (e.g., GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
input_image = input_image.to(device)
location_data = location_data.to(device)

# Run the model
with torch.no_grad():  # Disable gradient calculation for inference
    predicted_actions = model(input_image, location_data)

# Print the predicted actions
print(predicted_actions)
