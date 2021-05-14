# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 16:29:38 2020

@author: sihoon
"""

import torch
import torch.nn as nn


class HuberLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.SmoothL1Loss()

    def forward(self, logits, labels):
        loss = self.loss(logits, labels)
        return loss