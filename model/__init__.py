import torch.nn as nn
from .loss import SSIM, VGGLoss, PSNR
from .model import UNet, DnCNN, CascadedDnCNNWithUNet, ResidualDnCNN, DRUNet


def get_loss(name, **kwargs):
    name = name.lower()
    if name == 'mse':
        return nn.MSELoss()
    if name == 'ssim':
        return SSIM(**kwargs)
    if name == 'psnr':
        return PSNR()
    if name == 'vgg':
        return VGGLoss()


def get_model(**kwargs):
    model_name = kwargs['architecture'].lower()
    if model_name == 'unet':
        return UNet(**kwargs['UNet'])
    elif model_name == 'dncnn':
        return DnCNN(**kwargs['DnCNN'])
    elif model_name == 'cascaded':
        return CascadedDnCNNWithUNet(**kwargs['Cascaded'])
    elif model_name == 'residualdcnn':
        return ResidualDnCNN(**kwargs['ResidualDnCNN'])
    elif model_name == 'drunet':
        return DRUNet(**kwargs['DRUNet'])
