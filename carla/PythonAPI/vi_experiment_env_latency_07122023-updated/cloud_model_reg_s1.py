import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn as nn
import numpy as np
import timm
from torchvision.models.feature_extraction import create_feature_extractor
# # Load the pretrained ResNet-18 model


# # Create a new model without these layers
class CustomResNet18regs1(nn.Module):
    def __init__(self):
        super(CustomResNet18regs1, self).__init__()
        #self.resnet34 = timm.create_model('resnet34d',pretrained=True)
        #self.feature_extractor=nn.Sequential(*list(self.resnet34.children())[:-1])
        #print(self.resnet34.fc.in_features)
        self.model = timm.create_model('regnety_002', pretrained=True)
        #print(self.model.head.fc.in_features/2)
        # self.features=nn.Sequential(*list(self.model.children())[:-1])
        #print(self.model)
        self.model.s2 = nn.Identity()
        self.model.s3 = nn.Identity()
        self.model.s4 = nn.Identity()
        self.goal=nn.Sequential(
            nn.Linear(2, 12),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(12,24)
        )
        #print(self.model.forward_features)
        self.lin =nn.Sequential(
            nn.Linear(48, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 2)
        )
    def forward(self, x,locations,return_features=False):
         # Reshape the input from (batch_size, height, width, channel) to (batch_size, channel, height, width)
        #print(x.shape)
        # x=self.model.stem(x)
        # #print(x.shape)
        # x=self.model.s1(x)
        # with torch.no_grad():
        x = self.model.stem(x)
        x = self.model.s1(x)
        # x=self.model.s2(x)
        # x=self.model.s3(x)
        # x=self.model.s4(x)
        x=self.model.final_conv(x)
        x=self.model.head.global_pool(x)
        y=self.goal(locations)
        # print(x.shape)
        sf=torch.cat((x, y), dim=1)
        if return_features:
            return sf 
        sf = self.lin(sf)
        return sf