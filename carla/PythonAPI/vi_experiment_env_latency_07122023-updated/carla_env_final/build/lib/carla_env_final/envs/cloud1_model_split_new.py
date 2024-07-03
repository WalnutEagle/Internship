import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
import numpy as np
import timm
from torchvision.models.feature_extraction import create_feature_extractor




class CustomResNet18(nn.Module):
	def __init__(self):
		super(CustomResNet18, self).__init__()
		self.resnet34 = timm.create_model('resnet34d',pretrained=True)
		self.feature_extractor=nn.Sequential(*list(self.resnet34.children())[:-1])
		#print(self.resnet34.fc.in_features)
		self.resnet34.fc =nn.Sequential(
			nn.Linear(self.resnet34.fc.in_features+4, 512),
			nn.LeakyReLU(negative_slope=0.2),
			nn.Linear(512, 256),
			nn.LeakyReLU(negative_slope=0.2),
			nn.Linear(256, 2)
		)
	def forward(self, x,locations, split, env):
		 # Reshape the input from (batch_size, height, width, channel) to (batch_size, channel, height, width)
		#print(x.shape)
		
		modules = list(self.resnet34.children())[:-1]
		
		if env == "mobile":
			if split == len(modules)+1:
				x=self.feature_extractor(x)
				x=torch.cat((x, locations), dim=1)
				x = self.resnet34.fc(x)
				return x
			else:
				x = nn.Sequential(*modules[:split])(x)
				return x
		elif env == "cloud":
			x = nn.Sequential(*modules[split:])(x)
			x=torch.cat((x, locations), dim=1)
			x = self.resnet34.fc(x)
			return x
			
			

model = CustomResNet18()

dummy_images = torch.randn(1, 3, 224, 224)  # Example input image batch
dummy_locations = torch.randn(1, 4)  # Example locations data
dummy_targets = torch.tensor([[0, 1]], dtype=torch.float32)  # Example target

#criterion = nn.MSELoss()
#optimizer = optim.Adam(model.parameters(), lr=0.001)
#outputs = model(dummy_images, dummy_locations)
#loss = criterion(outputs, dummy_targets)

#optimizer.zero_grad()
#loss.backward()
#optimizer.step()

#print(f"Loss: {loss.item()}")
#torch.save(model.state_dict(), 'dummy_weights.pth')


model = CustomResNet18()
model2 = CustomResNet18()

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

