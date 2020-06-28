# -*- coding: utf-8 -*-
import torch.nn as nn
from . import resnet

class Network(nn.Module):
    def __init__(self, cfg):
        super(Network, self).__init__()
        self.resnet = resnet.Resnet(cfg)


    def forward(self, images, target=None):
        # TODO：model1
        # # 输入变为image_list
        feature, x = self.resnet(images)

        if self.training:
            losses = {}
            return losses, x

        return feature, x
