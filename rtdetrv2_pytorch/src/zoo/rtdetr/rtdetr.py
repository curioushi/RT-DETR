"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np 
from typing import List 

from ...core import register


__all__ = ['RTDETR', ]


@register()
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, \
        backbone: nn.Module, 
        encoder: nn.Module, 
        decoder: nn.Module, 
    ):
        super().__init__()
        self.backbone = backbone

        # # replace the first conv layer with a 4-channel conv layer
        # original_first_conv = self.backbone.conv1.conv1_1.conv
        # new_first_conv = nn.Conv2d(4, 32, kernel_size=(3,3), stride=(2,2), padding=(1,1), bias=False)
        # with torch.no_grad():
        #     torch.nn.init.kaiming_normal_(new_first_conv.weight, mode='fan_out', nonlinearity='relu')
        #     new_first_conv.weight *= 0.001
        #     new_first_conv.weight[:, :3, :, :] = original_first_conv.weight.data.clone()
        # self.backbone.conv1.conv1_1.conv = new_first_conv

        self.decoder = decoder
        self.encoder = encoder
        
    def forward(self, x, sparse_xyzs, targets=None):
        x = self.backbone(x)
        x_high_res, x = x[0], x[1:]
        x = self.encoder(x)
        x = self.decoder(x, x_high_res, sparse_xyzs, targets)

        return x
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 
