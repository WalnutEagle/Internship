import torch
import torch.nn as nn
import torch.optim as optim
import timm

class CustomResNet18(nn.Module):
    def __init__(self):
        super(CustomResNet18, self).__init__()
        #self.resnet34 = timm.create_model('resnet34d',pretrained=True)
        #self.feature_extractor=nn.Sequential(*list(self.resnet34.children())[:-1])
        #print(self.resnet34.fc.in_features)
        self.model = timm.create_model('regnety_002', pretrained=True)
        print(self.model.head.fc.in_features/2)
        self.goal=nn.Sequential(
            nn.Linear(2, int(self.model.head.fc.in_features/2)),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(int(self.model.head.fc.in_features/2),int(self.model.head.fc.in_features))
        )
        #print(self.model.forward_features)
        self.lin =nn.Sequential(
            nn.Linear(self.model.head.fc.in_features*2, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 2)
        )
        
        self.layers = [
            self.model.stem,
            self.model.s1,
            self.model.s2,
            self.model.s3,
            self.model.s4,
            self.model.final_conv,
            self.model.head.global_pool,
            lambda x: torch.cat((x, self.goal(locations)), dim=1),
            self.lin
        ]
        
    def forward(self, x, locations, split=None, env="mobile"):
        if env == "mobile":
            for i, layer in enumerate(self.layers):
                if callable(layer):
                    x = layer(x)
                else:
                    x = layer(x)
                if split is not None and i == split - 1:
                    return x
        elif env == "cloud":
            for i, layer in enumerate(self.layers[split:], start=split):
                if callable(layer):
                    x = layer(x)
                else:
                    x = layer(x)
        return x



#model = CustomResNet18()

# Sample input (adjust size as per your requirements)
#x = torch.randn(1, 3, 224, 224)  # Example input tensor (batch_size, channels, height, width)
#locations = torch.randn(1, 2)  # Example additional input for 'locations'

# If you have a specific target and loss function
#target = torch.randn(1, 2)  # Example target
#criterion = nn.MSELoss()

# Define an optimizer
#optimizer = optim.Adam(model.parameters(), lr=0.001)

# Forward pass
#output = model(x, locations)
#loss = criterion(output, target)
#optimizer.zero_grad()
#loss.backward()
#optimizer.step()
#torch.save(model.state_dict(), 'custom_resnet18_weights.pth')


# model = CustomResNet18()

# # Load the saved weights
# model.load_state_dict(torch.load('custom_resnet18_weights.pth'))

# # Ensure the model is in evaluation mode
# model.eval()

# # Sample input (adjust size as per your requirements)
# x = torch.randn(1, 3, 224, 224)  # Example input tensor (batch_size, channels, height, width)
# locations = torch.randn(1, 2)  # Example additional input for 'locations'

# # Forward pass
# with torch.no_grad():
#     split_layer = 4
#     output = model(x, locations,split_layer,"mobile")
#     output = model(output, locations,split_layer,"cloud")

# # Process the output as needed
# print(output)
