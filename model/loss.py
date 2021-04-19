import torch
import torch.nn.functional as F
import torch.nn as nn
from .layers import Vgg19
from torch.distributions.multivariate_normal import MultivariateNormal
from itertools import product


def create_kernel(window_size, channel=1, dim=2):
    mu = torch.tensor([float(window_size) // 2] * dim)
    sigma = 1.5
    if dim == 2:
        sigma = torch.tensor([[sigma, 0], [0, sigma]])
    elif dim == 3:
        sigma = torch.tensor([[sigma, 0, 0], [0, sigma, 0], [0, 0, sigma]])
    else:
        raise ValueError("Invalid dim")
    gaussian_dist = MultivariateNormal(mu, sigma)
    kernel = torch.zeros((window_size,) * dim)
    iterator = product(*(range(window_size),) * dim)
    for c in iterator:
        kernel[c] = gaussian_dist.log_prob(torch.tensor(c)).exp()
    kernel /= kernel.sum()
    kernel = kernel.expand((channel, 1) + (window_size,) * dim).contiguous()
    return kernel


def _ssim(t1, t2, window, window_size, channel, size_average=True):
    assert len(t1.shape) == len(t2.shape)
    conv_op = F.conv2d if len(t1.shape) == 4 else F.conv3d
    mu1 = conv_op(t1, window, padding=window_size // 2, groups=channel)
    mu2 = conv_op(t2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = conv_op(t1 * t1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = conv_op(t2 * t2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = conv_op(t1 * t2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(nn.Module):
    def __init__(self, window_size=11, channel=1, dim=2, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.dim = dim
        self.window = create_kernel(self.window_size, self.channel, self.dim)

    def forward(self, img1, img2):
        assert img1.device == img2.device
        channel1 = img1.size(1)
        channel2 = img2.size(1)
        assert self.channel == channel1 == channel2
        if img1.device != self.window.device:
            self.window = self.window.to(img1.device)
        return _ssim(img1, img2, self.window, self.window_size, self.channel, self.size_average)


def ssim(img1, img2, window_size=11, dim=2, size_average=True):
    assert img1.device == img2.device
    channel = img1.size(1)
    window = create_kernel(window_size, channel, dim).type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)


class PSNR(nn.Module):
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""
    def __init__(self):
        super(PSNR, self).__init__()

    def forward(self, img1, img2):
        return psnr(img1, img2)


def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(40.0 / torch.sqrt(mse))


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)

    def forward(self, x, y):
        while x.size()[3] > 1024:
            x, y = self.downsample(x), self.downsample(y)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss
