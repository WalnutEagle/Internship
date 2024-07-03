import torch
import torch.nn as nn
import numpy as np
import timm



class ClassificationMobileVitXSNetwork(torch.nn.Module):
    def __init__(self):
        """
        Implementation of the network layers. The image size of the input
        observations is 96x96 pixels.
        """
        super(ClassificationMobileVitXSNetwork, self).__init__()
        #self.gpu = torch.device('cuda')
        #self.device = torch.device('cuda')
        #Implemented a 3 convolution and 2 linear layer network
        self.mobile = timm.create_model('mobilevit_xs',pretrained=True)
        self.feature_extractor=nn.Sequential(*list(self.mobile.children())[:-1],self.mobile.head.global_pool, self.mobile.head.drop)
        print(self.feature_extractor)
        self.lin =nn.Sequential(
            nn.Linear(self.mobile.head.in_features*2, 2))
        self.goal=nn.Linear(2, self.mobile.head.in_features)

    def forward(self, observation, locations):
        """
        The forward pass of the network. Returns the prediction for the given
        input observation.
        observation:   torch.Tensor of size (batch_size, height, width, channel)
        return         torch.Tensor of size (batch_size, C)
        """
        x=self.feature_extractor(observation)
        # print(x.shape)
        y=self.goal(locations)
        ft=torch.cat((x, y), dim=1)
        ft = self.lin(ft)
        #return nn.functional.softmax(x, dim=1)
        return ft