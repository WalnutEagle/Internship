import torch
import torch.nn as nn
import numpy as np
import timm


class ClassificationNetwork(torch.nn.Module):
    def __init__(self):
        """
        Implementation of the network layers. The image size of the input
        observations is 96x96 pixels.
        """
        super(ClassificationNetwork, self).__init__()
        #self.gpu = torch.device('cuda')
        #self.device = torch.device('cuda')
        #Implemented a 3 convolution and 2 linear layer network
        self.mobile = timm.create_model('mobilenetv2_100',pretrained=True)
        self.feature_extractor=nn.Sequential(*list(self.mobile.children())[:-1])
        self.mobile.classifier =nn.Sequential(
            nn.Linear(self.mobile.classifier.in_features+4, 2))

    def forward(self, observation, locations):
        """
        The forward pass of the network. Returns the prediction for the given
        input observation.
        observation:   torch.Tensor of size (batch_size, height, width, channel)
        return         torch.Tensor of size (batch_size, C)
        """
        x=self.feature_extractor(observation)
        # print(x.shape)
        x=torch.cat((x, locations), dim=1)
        x = self.mobile.classifier(x)
        #return nn.functional.softmax(x, dim=1)
        return x