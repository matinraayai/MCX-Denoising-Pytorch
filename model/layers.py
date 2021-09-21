import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    The original res block as described in the resnet paper.
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, do_3d=False,
                 kernel_size=3, stride=1, dilation=1,
                 padding=1,
                 padding_mode='replicate',
                 projection=True,
                 activation_fn=F.relu):
        super(ResidualBlock, self).__init__()
        conv_layer = nn.Conv3d if do_3d else nn.Conv2d
        norm_layer = nn.BatchNorm3d if do_3d else nn.BatchNorm2d
        self.conv1 = conv_layer(in_channels, out_channels,
                                kernel_size=kernel_size, stride=stride,
                                padding=padding, padding_mode=padding_mode,
                                bias=False,
                                dilation=dilation)
        self.bn1 = norm_layer(out_channels)
        self.conv2 = conv_layer(out_channels, out_channels,
                                kernel_size=kernel_size, stride=stride,
                                padding=padding, padding_mode=padding_mode,
                                bias=False,
                                dilation=dilation)
        self.bn2 = norm_layer(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            if projection:
                self.shortcut = nn.Sequential(conv_layer(in_channels, out_channels, kernel_size=1, padding=0),
                                              norm_layer(out_channels))
            else:
                self.shortcut = lambda x: x
        self.activation_fn = activation_fn

    def forward(self, x):
        out = self.activation_fn(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation_fn(out)
        return out


class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        from torchvision import models
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
