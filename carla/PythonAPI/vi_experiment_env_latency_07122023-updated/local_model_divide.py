# import torch
# import torch.nn as nn
# import numpy as np
# import timm


# class ClassificationNetworkDivide(torch.nn.Module):
#     def __init__(self):
#         """
#         Implementation of the network layers. The image size of the input
#         observations is 96x96 pixels.
#         """
#         super(ClassificationNetworkDivide, self).__init__()
#         #self.gpu = torch.device('cuda')
#         #self.device = torch.device('cuda')
#         #Implemented a 3 convolution and 2 linear layer network
#         self.mobile = timm.create_model('mobilenetv2_100',pretrained=True)
#         self.feature_extractor=nn.Sequential(*list(self.mobile.children())[:-4])
#         print(self.feature_extractor)
#         #print(self.feature_extractor)
#         # self.lin =nn.Sequential(
#         #     nn.Linear(self.mobile.classifier.in_features*2, 2))
#         self.l_1 = self.mobile.conv_head
#         self.l_1.out_channels=320
#         self.l_2= self.mobile.bn2
#         self.l_2.num_features=320
#         self.l_3 = self.mobile.global_pool
#         self.g_1 = self.mobile.conv_head
#         self.g_2= self.mobile.bn2
#         self.g_3 = self.mobile.global_pool
#         self.goal_l=nn.Linear(2, 320)
#         self.goal_g=nn.Linear(2, 1280)
#         self.lin_l=nn.Linear(320*2, 2)
#         self.lin_g=nn.Linear(1280*2, 2)

#     def forward(self, observation, locations):
#         """
#         The forward pass of the network. Returns the prediction for the given
#         input observation.
#         observation:   torch.Tensor of size (batch_size, height, width, channel)
#         return         torch.Tensor of size (batch_size, C)
#         """
#         x=self.feature_extractor(observation)
#         xl=self.l_1(x)
#         xl=self.l_2(xl)
#         xl=self.l_3(xl)
#         # print(x.shape)
#         xg=self.g_1(x)
#         xg=self.g_2(xg)
#         xg=self.g_3(xg)
#         yl=self.goal_l(locations)
#         yg=self.goal_g(locations)
#         print(yl.shape)
#         print(xl.shape)
#         ftl=torch.cat((xl, yl), dim=1)
#         ftg=torch.cat((xg, yg), dim=1)
#         ftl = self.lin_l(ftl)
#         ftg = self.lin_g(ftg)
#         #return nn.functional.softmax(x, dim=1)
#         return ftl,ftg