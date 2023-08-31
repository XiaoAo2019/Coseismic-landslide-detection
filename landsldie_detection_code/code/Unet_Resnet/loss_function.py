########################################################################################################################
# loss函数
########################################################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn

def build_target(target: torch.Tensor, num_classes: int = 2, ignore_index: int = -100):
    """build target for dice coefficient"""
    dice_target = target.clone()
    # if ignore_index >= 0:
    ignore_mask = torch.eq(target, ignore_index)
    dice_target[ignore_mask] = 2
    # [N, H, W] -> [N, H, W, C]
    dice_target = nn.functional.one_hot(dice_target, num_classes+1).float()
    # else:
    #     dice_target = nn.functional.one_hot(dice_target, num_classes).float()

    return dice_target

def CE_Loss(inputs, target, ignore_index: int = -100):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    CE_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)(temp_inputs, temp_target)
    return CE_loss

def Focal_Loss(inputs, target, ignore_index: int = -100, alpha=0.5, gamma=2):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    logpt = -nn.CrossEntropyLoss(ignore_index=ignore_index)(temp_inputs, temp_target)

    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt
    focal_loss = loss.mean()
    return focal_loss

def Dice_loss(inputs, dice_target, beta=1, smooth=1e-5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = dice_target.size()

    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = dice_target.view(n, -1, ct)

    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
    fn = torch.sum(temp_target[...,:-1], axis=[0, 1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss


