import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn as nn
import numpy as np
import timm
from torchvision.models.feature_extraction import create_feature_extractor

			

model = CustomResNet18()

dummy_images = torch.randn(1, 3, 224, 224)  # Example input image batch
dummy_locations = torch.randn(1, 4)  # Example locations data
dummy_targets = torch.tensor([[0, 1]], dtype=torch.float32)  # Example target

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
outputs = model(dummy_images, dummy_locations)
loss = criterion(outputs, dummy_targets)

optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"Loss: {loss.item()}")
torch.save(model.state_dict(), 'dummy_weights.pth')


weights_path = r"dummy_weights.pth"
state_dict = torch.load(weights_path)

#model = torch.load(weights_path)
model.load_state_dict(state_dict)
model.eval()

model2.load_state_dict(state_dict)
model2.eval()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

dummy_images = dummy_images.to(device)
dummy_locations = dummy_locations.to(device)

split = 5

# Perform inference
with torch.no_grad():
	output = model(dummy_images, dummy_locations, split, "mobile")
	print(output.shape)
	output = model(output, dummy_locations, split, "cloud")

print(output.shape)
