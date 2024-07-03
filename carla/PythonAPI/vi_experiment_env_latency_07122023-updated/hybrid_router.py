import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical



class Hybrid_Router(nn.Module):
    def __init__(self, n_labels=2, num_features=1280):
        # print('---------------------------------- Hybrid Router (student_ft) --')
        super(Hybrid_Router, self).__init__()
        
        self.n_labels = n_labels
        n_ft = 64
        n_ft2 = 128

        self.transform_ft = nn.Sequential(
                nn.Linear(num_features, n_ft2, bias=True),
                nn.BatchNorm1d( n_ft2 ),
                nn.ReLU(),
                nn.Linear(n_ft2, n_ft, bias=True),
                nn.BatchNorm1d( n_ft ),
                nn.ReLU(),
            )

        logit_ft_num = 2
        self.routing = nn.Sequential(
                nn.Linear(logit_ft_num + n_ft, n_labels, bias=True),
                nn.BatchNorm1d( n_labels ),
                nn.ReLU(),
                nn.Linear(n_labels, n_ft, bias=True),
                nn.BatchNorm1d( n_ft ),
                nn.ReLU(),
                nn.Linear(n_ft, n_labels, bias=True),
                nn.BatchNorm1d( n_labels ),
                nn.ReLU(),
                nn.Linear(n_labels, 1), # bias=True),
                nn.Sigmoid(),
            )

    def forward(self, s_logits, s_ft ):
        s_ft = self.transform_ft( s_ft )
        x = torch.cat([ s_logits, s_ft ], dim=1)
        return self.routing( x )