# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import functools

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F
import torchvision

logger = logging.getLogger(__name__)


class FCN(nn.Module):
    def __init__(self, config, **kwargs):
        extra = config.MODEL.EXTRA
        super().__init__()

        fcn = torchvision.models.segmentation.fcn_resnet101(config.MODEL.PRETRAINED)
        self.backbone = fcn.backbone
        n_classes = config.DATASET.NUM_CLASSES
        in_ch = 2048
        inter_ch = 512
        self.classifier = nn.Sequential(
            nn.Conv2d(in_ch, inter_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_ch),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_ch, n_classes, 1),
        )

        self.init_weights_module(self.classifier.modules())

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x['out'])
        return x

    def init_weights_module(self, modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
 
    def init_weights(self, pretrained='',):
        logger.info('=> init weights from normal distribution')
        self.init_weights_module(self.modules())
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            #for k, _ in pretrained_dict.items():
            #    logger.info(
            #        '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

def get_seg_model(cfg, **kwargs):
    model = FCN(cfg, **kwargs)

    return model

if __name__ == '__main__':
    model = get_fcn_model({})
    print(model)


