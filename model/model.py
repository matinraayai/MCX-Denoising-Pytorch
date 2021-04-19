"""
Contains all the de-noising models used in literature and new ones proposed by us.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .layers import ResidualBlock


# TODO: Despite not being of use for now, play around with the parameters and instantiations to see if we get a better
# result in the final model
# For now, do not use this.
def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    if isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


class UNet(nn.Module):
    def __init__(self, do_3d=False, activation_fn='F.relu', **kwargs):
        super(UNet, self).__init__()
        conv_layer = nn.Conv3d if do_3d else nn.Conv2d
        norm_layer = nn.BatchNorm3d if do_3d else nn.BatchNorm2d
        pool_layer = nn.MaxPool3d if do_3d else nn.MaxPool2d
        conv_trans_layer = nn.ConvTranspose3d if do_3d else nn.ConvTranspose2d
        self.activation_fn = eval(activation_fn)
        # Down0
        self.down0a = conv_layer(1, 64, 3, padding=1, padding_mode='replicate')
        self.down0a_norm = norm_layer(64)
        self.down0b = conv_layer(64, 64, 3, padding=1, padding_mode='replicate')
        self.down0b_norm = norm_layer(64)
        self.down0c = pool_layer(2)
        # Down1
        self.down1a = conv_layer(64, 128, 3, padding=1, padding_mode='replicate')
        self.down1a_norm = norm_layer(128)
        self.down1b = conv_layer(128, 128, 3, padding=1, padding_mode='replicate')
        self.down1b_norm = norm_layer(128)
        self.down1c = pool_layer(2)
        # Down2
        self.down2a = conv_layer(128, 256, 3, padding=1, padding_mode='replicate')
        self.down2a_norm = norm_layer(256)
        self.down2b = conv_layer(256, 256, 3, padding=1, padding_mode='replicate')
        self.down2b_norm = norm_layer(256)
        # Up1
        self.up1a = conv_trans_layer(256, 128, 2, 2)
        self.up1a_norm = norm_layer(128)
        self.up1c = conv_layer(256, 128, 3, padding=1, padding_mode='replicate')
        self.up1c_norm = norm_layer(128)
        self.up1d = conv_layer(128, 128, 3, padding=1, padding_mode='replicate')
        self.up1d_norm = norm_layer(128)
        self.up1e = conv_layer(128, 128, 3, padding=1, padding_mode='replicate')
        self.up1e_norm = norm_layer(128)
        # Up0
        self.up0a = conv_trans_layer(128, 64, 2, 2)
        self.up0a_norm = norm_layer(64)
        self.up0c = conv_layer(128, 64, 3, padding=1, padding_mode='replicate')
        self.up0c_norm = norm_layer(64)
        self.up0d = conv_layer(64, 64, 3, padding=1, padding_mode='replicate')
        self.up0d_norm = norm_layer(64)
        self.up0e = conv_layer(64, 64, 3, padding=1, padding_mode='replicate')
        self.up0e_norm = norm_layer(64)
        self.last_layer = conv_layer(64, 1, 1)

    def forward(self, x):
        down0a = self.activation_fn(self.down0a_norm(self.down0a(x)))
        down0b = self.activation_fn(self.down0b_norm(self.down0b(down0a)))
        down0c = self.down0c(down0b)
        down1a = self.activation_fn(self.down1a_norm(self.down1a(down0c)))
        down1b = self.activation_fn(self.down1b_norm(self.down1b(down1a)))
        down1c = self.down1c(down1b)
        down2a = self.activation_fn(self.down2a_norm(self.down2a(down1c)))
        down2b = self.activation_fn(self.down2b_norm(self.down2b(down2a)))
        up1a = self.up1a_norm(self.up1a(down2b))
        up1b = torch.cat([up1a, down1b], dim=1)
        up1c = self.activation_fn(self.up1c_norm(self.up1c(up1b)))
        up1d = self.activation_fn(self.up1d_norm(self.up1d(up1c)))
        up1e = self.activation_fn(self.up1e_norm(self.up1e(up1d)))
        up0a = self.up0a_norm(self.up0a(up1e))
        up0b = torch.cat([up0a, down0b], dim=1)
        up0c = self.activation_fn(self.up0c_norm(self.up0c(up0b)))
        up0d = self.activation_fn(self.up0d_norm(self.up0d(up0c)))
        up0e = self.activation_fn(self.up0e_norm(self.up0e(up0d)))
        output = self.last_layer(up0e)
        return x - output


class DnCNN(nn.Module):
    def __init__(self, do_3d=False, kernel_size=3, padding=1, padding_mode='replicate', input_channels=1,
                 output_channels=1,
                 inter_kernel_channel=64, num_layers=17, activation_fn='F.relu'):
        super(DnCNN, self).__init__()
        self.num_layers = num_layers
        self.activation_fn = eval(activation_fn)
        conv_layer = nn.Conv3d if do_3d else nn.Conv2d
        norm_layer = nn.BatchNorm3d if do_3d else nn.BatchNorm2d
        self.__setattr__("conv1", conv_layer(input_channels,
                                             inter_kernel_channel,
                                             kernel_size,
                                             padding_mode=padding_mode,
                                             padding=padding))
        for num_layer in range(2, num_layers):
            self.__setattr__(f"conv{num_layer}", conv_layer(inter_kernel_channel,
                                                            inter_kernel_channel,
                                                            kernel_size,
                                                            padding_mode=padding_mode,
                                                            padding=padding,
                                                            bias=False))
            self.__setattr__(f"norm{num_layer}", norm_layer(inter_kernel_channel))
        self.__setattr__(f"conv{num_layers}", conv_layer(inter_kernel_channel,
                                                         output_channels,
                                                         kernel_size,
                                                         padding_mode=padding_mode,
                                                         padding=padding))

    def forward(self, x):
        output = self.activation_fn(self.conv1(x))
        for num_layer in range(2, self.num_layers):
            output = self.__getattr__(f"conv{num_layer}")(output)
            output = self.__getattr__(f"norm{num_layer}")(output)
            output = self.activation_fn(output)
        output = self.__getattr__(f"conv{self.num_layers}")(output)
        return x - output


class ResidualDnCNN(nn.Module):
    def __init__(self, do_3d=False, in_channels=1, out_channels=1,
                 inter_kernel_channel=128,
                 kernel_size=3,
                 padding_mode='replicate',
                 padding=1,
                 num_layers=17, activation_fn='nn.ReLU(inplace=True)'):
        super(ResidualDnCNN, self).__init__()
        self.num_layers = num_layers
        self.activation_fn = eval(activation_fn)
        conv_layer = nn.Conv3d if do_3d else nn.Conv2d
        self.conv1 = conv_layer(in_channels, inter_kernel_channel, kernel_size=kernel_size,
                                padding_mode=padding_mode, padding=padding)
        self.residual_blocks = []
        for num_layer in range(2, num_layers):
            self.residual_blocks.append(ResidualBlock(in_channels=inter_kernel_channel,
                                                      out_channels=inter_kernel_channel,
                                                      do_3d=do_3d,
                                                      kernel_size=kernel_size,
                                                      padding_mode=padding_mode,
                                                      padding=padding,
                                                      dilation=1,
                                                      activation_fn=self.activation_fn))
        self.residual_blocks = nn.Sequential(*self.residual_blocks)
        self.__setattr__(f"conv{num_layers}", conv_layer(inter_kernel_channel, out_channels,
                                                         kernel_size,
                                                         padding_mode=padding_mode, padding=padding))

    def forward(self, x):
        output_initial = self.activation_fn(self.conv1(x))
        output = self.residual_blocks(output_initial)
        output += output_initial
        output = self.__getattr__(f"conv{self.num_layers}")(output)
        return output


class CascadedDnCNNWithUNet(nn.Module):
    def __init__(self, do_3d=False, num_dncnn=1, output_channels=1, num_dncnn_layers=17, activation_fn='F.relu'):
        super(CascadedDnCNNWithUNet, self).__init__()
        self.num_dncnn = num_dncnn
        for num in range(self.num_dncnn):
            self.__setattr__(f"dncnn{num}", DnCNN(output_channels=output_channels, do_3d=do_3d,
                                                  num_layers=num_dncnn_layers, activation_fn=activation_fn))
        self.unet = UNet(do_3d=do_3d)

    def forward(self, x):
        for num in range(self.num_dncnn):
            x = self.__getattr__(f"dncnn{num}")(x)
        return self.unet(x)


class DRUNet(nn.Module):
    def __init__(self, in_nc=1, out_nc=1,
                 res_block_channels=None,
                 num_res_blocks=4, activation_function='F.relu'):
        super(DRUNet, self).__init__()
        if res_block_channels is None:
            res_block_channels = [64, 128, 256, 512]
        self.m_head = nn.Conv2d(in_nc, res_block_channels[0], bias=False, kernel_size=3, stride=1, padding=1)

        activation_function = eval(activation_function)
        self.activation_fn = activation_function
        def downsample_block(in_channels, out_channels):
            module = [ResidualBlock(in_channels, in_channels, activation_fn=activation_function)
                      for _ in range(num_res_blocks)]
            module += [nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False)]
            return nn.Sequential(*module)

        self.m_down1 = downsample_block(res_block_channels[0], res_block_channels[1])

        self.m_down2 = downsample_block(res_block_channels[1], res_block_channels[2])

        self.m_down3 = downsample_block(res_block_channels[2], res_block_channels[3])

        self.m_body = [ResidualBlock(res_block_channels[3], res_block_channels[3]) for _ in range(num_res_blocks)]

        self.m_body = nn.Sequential(*self.m_body)

        def upsample_block(in_channels, out_channels):
            module = [nn.ConvTranspose2d(in_channels, out_channels, bias=False, kernel_size=2, stride=2)]
            module += [nn.ReLU(inplace=True)]
            module += [ResidualBlock(out_channels, out_channels, activation_fn=activation_function)
                       for _ in range(num_res_blocks)]
            return nn.Sequential(*module)

        self.m_up3 = upsample_block(res_block_channels[3], res_block_channels[2])

        self.m_up2 = upsample_block(res_block_channels[2], res_block_channels[1])

        self.m_up1 = upsample_block(res_block_channels[1], res_block_channels[0])

        self.m_tail = nn.Conv2d(res_block_channels[0], out_nc, bias=False, kernel_size=3, stride=1, padding=1)

    def forward(self, x0):
        x1 = self.activation_fn(self.m_head(x0))
        x2 = self.activation_fn(self.m_down1(x1))
        x3 = self.activation_fn(self.m_down2(x2))
        x4 = self.activation_fn(self.m_down3(x3))
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)
        return x
