"""
Initialization schemes for models.
Borrowed from Borrowed from https://github.com/zudi-lin/pytorch_connectomics/.
"""
import torch.nn as nn
from math import sqrt


def init_weights(model, policy):
    if policy == "xavier":
        xavier_init(model)
    elif policy == "he":
        he_init(model)
    elif policy == "selu":
        selu_init(model)
    elif policy == "ortho":
        ortho_init(model)


def xavier_init(model):
    """
    Default Xavier initialization
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
            nn.init.xavier_uniform(m.weight, gain=nn.init.calculate_gain('relu'))


def he_init(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
            nn.init.kaiming_normal(m.weight, mode='fan_in')


def selu_init(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            fan_in = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
            nn.init.normal(m.weight, 0, sqrt(1. / fan_in))
        elif isinstance(m, nn.Linear):
            fan_in = m.in_features
            nn.init.normal(m.weight, 0, sqrt(1. / fan_in))


def ortho_init(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
            nn.init.orthogonal_(m.weight)
