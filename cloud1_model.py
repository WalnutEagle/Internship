# import torch
# import torch.nn as nn
# import torchvision.models as models
# import torch.nn as nn
# import numpy as np
# import timm
# from torchvision.models.feature_extraction import create_feature_extractor
# # Load the pretrained ResNet-18 model


# # # Create a new model without these layers
# class CustomResNet18(nn.Module):
#     def __init__(self):
#         super(CustomResNet18, self).__init__()
#         self.resnet34 = timm.create_model('resnet34d',pretrained=True)
#         self.feature_extractor=nn.Sequential(*list(self.resnet34.children())[:-1])
#         #print(self.resnet34.fc.in_features)
#         self.resnet34.fc =nn.Sequential(
#             nn.Linear(self.resnet34.fc.in_features+2, 512),
#             nn.LeakyReLU(negative_slope=0.2),
#             nn.Linear(512, 256),
#             nn.LeakyReLU(negative_slope=0.2),
#             nn.Linear(256, 2)
#         )
#     def forward(self, x,locations):
#          # Reshape the input from (batch_size, height, width, channel) to (batch_size, channel, height, width)
#         #print(x.shape)

#         x=self.feature_extractor(x)
#         # print(x.shape)
#         x=torch.cat((x, locations), dim=1)
#         x = self.resnet34.fc(x)
#         return x

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn as nn
import numpy as np
import timm
from torchvision.models.feature_extraction import create_feature_extractor
# # Load the pretrained ResNet-18 model


# # Create a new model without these layers
class CustomResNet18(nn.Module):
    def __init__(self):
        super(CustomResNet18, self).__init__()
        #self.resnet34 = timm.create_model('resnet34d',pretrained=True)
        #self.feature_extractor=nn.Sequential(*list(self.resnet34.children())[:-1])
        #print(self.resnet34.fc.in_features)
        self.model = timm.create_model('regnety_002', pretrained=True)
        #print(self.model.head.fc.in_features/2)
        self.features=nn.Sequential(*list(self.model.children())[:-1])
        print(self.model)
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
    def forward(self, x,locations):
         # Reshape the input from (batch_size, height, width, channel) to (batch_size, channel, height, width)
        #print(x.shape)
        x=self.model.stem(x)
        #print(x.shape)
        x=self.model.s1(x)
        x=self.model.s2(x)
        x=self.model.s3(x)
        x=self.model.s4(x)
        x=self.model.final_conv(x)
        x=self.model.head.global_pool(x)
        y=self.goal(locations)
        # print(x.shape)
        sf=torch.cat((x, y), dim=1)
        sf = self.lin(sf)
        return sf






