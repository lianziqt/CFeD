#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    def __init__(self):
        super(DistillationLoss, self).__init__()

    def forward(self, output, old_target, temperature, frac):
        T = temperature
        alpha = frac
        outputs_S = F.log_softmax(output / T, dim=1)
        outputs_T = F.softmax(old_target / T, dim=1)
        l_old = outputs_T.mul(outputs_S)
        l_old = -1.0 * torch.sum(l_old) / outputs_S.shape[0]

        return l_old * alpha