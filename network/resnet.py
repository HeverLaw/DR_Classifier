# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torchvision import models

class StemWithBatchNorm(nn.Module):
    def __init__(self, cfg, resnet):
        super(StemWithBatchNorm, self).__init__()

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu1 = resnet.relu
        self.maxpool1 = resnet.maxpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        return x


class Resnet(nn.Module):
    def __init__(self, cfg):
        super(Resnet, self).__init__()
        # filters = [64, 128, 256, 512]
        if cfg.BACKBONE == 'resnet18':
            resnet = models.resnet18(pretrained=True, num_classes=cfg.NUM_FEATURES)
        else:
            resnet = models.resnet34(pretrained=True, num_classes=cfg.NUM_FEATURES)
        self.stem = StemWithBatchNorm(cfg, resnet)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        # resnet内部已初始化
        self.avgpool = resnet.avgpool
        # self.fc1 = nn.Linear(512, 5)
        self.fc1 = nn.Linear(512, cfg.NUM_CATEGORY)
        # TODO：做一个固定参数的功能
        self._freeze_backbone(cfg.FREEZE_CONV_BODY_AT)
        nn.init.kaiming_uniform_(self.fc1.weight, a=1)
        nn.init.constant_(self.fc1.bias, 0)


    def _freeze_backbone(self, freeze_at):
        # pass
        for stage_index in range(freeze_at):
            if stage_index == 0:
                m = self.stem  # stage 0 is the stem
            else:
                m = getattr(self, "layer" + str(stage_index))
            for p in m.parameters():
                p.requires_grad = False

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        feature = torch.flatten(x, 1)
        x = self.fc1(feature)
        return feature, x