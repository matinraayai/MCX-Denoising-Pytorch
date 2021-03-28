"""
YACS configs for training and inference
Borrowed from https://github.com/zudi-lin/pytorch_connectomics/
"""
import torch
import torch.nn as nn
from .model import UNet, CascadedDnCNNWithUNet, DRUNet, DnCNN, ResidualDnCNN
from .loss import SSIM, PSNR, VGGLoss


def get_regularizer(reg_opts=()):
    regularizers = []
    if reg_opts:
        for reg_opt in reg_opts:
            if reg_opt == 0:
                regularizers.append(nn.L1Loss())
    return nn.Sequential(*regularizers)

def get_model(**kwargs):
    model_name = kwargs['architecture'].lower()
    if model_name == 'unet':
        return UNet(**kwargs['UNet'])
    elif model_name == 'dncnn':
        return DnCNN(**kwargs['DnCNN'])
    elif model_name == 'cascaded':
        return CascadedDnCNNWithUNet(**kwargs['Cascaded'])
    elif model_name == 'residualdncnn':
        return ResidualDnCNN(**kwargs['ResidualDnCNN'])
    elif model_name == 'drunet':
        return DRUNet(**kwargs['DRUNet'])


def load_model_from_checkpoint(checkpoint_dir, **kwargs):
    model = get_model(**kwargs).cuda()
    state_dict = torch.load(checkpoint_dir)
    model.load_state_dict(state_dict.state_dict())
    return model


def get_loss(loss_opt=(), **kwargs):
    out = []
    for opt in loss_opt:
        opt = opt.lower()
        if opt == 'mse':
            out.append(nn.MSELoss())
        elif opt == 'mae':
            out.append(nn.L1Loss())
        elif opt == 'ssim':
            out.append(SSIM(kwargs['ssim']))
        elif opt == 'psnr':
            out.append(PSNR())
        elif opt == 'vggloss':
            out.append(VGGLoss())
        else:
            raise ValueError(f'Unknown loss option {opt}')
    return out


class Criterion(nn.Module):
    def __init__(self, loss_option=('MSE',), loss_weight=(1.,),
                 regularizer_opt=(), regularizer_weight=(), **kwargs):
        super().__init__()
        self.loss_weight = loss_weight
        self.reg_weight = regularizer_weight

        self.loss = get_loss(loss_option, **kwargs)
        self.reg = get_regularizer(regularizer_opt)

    def forward(self, pred, target):
        output = 0
        for loss, weight in zip(self.loss, self.loss_weight):
            output += weight * loss(pred, target)
        for reg, weight in zip(self.reg, self.reg_weight):
            output += reg(pred) * weight
        return output

