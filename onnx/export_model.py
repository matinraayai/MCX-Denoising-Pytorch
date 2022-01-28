
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import sys
sys.path.append('..')
from model.builder import load_model_from_lightning_checkpoint


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        padding_mode = 'reflect'
        conv_layer = nn.Conv3d
        norm_layer = nn.BatchNorm3d
        pool_layer = nn.MaxPool3d
        conv_trans_layer = nn.ConvTranspose3d
        self.activation_fn = F.relu
        # Down0
        self.down0a = conv_layer(1, 64, 3, padding=1, padding_mode=padding_mode)
        self.down0a_norm = norm_layer(64)
        self.down0b = conv_layer(64, 64, 3, padding=1, padding_mode=padding_mode)
        self.down0b_norm = norm_layer(64)
        self.down0c = pool_layer(2)
        # Down1
        self.down1a = conv_layer(64, 128, 3, padding=1, padding_mode=padding_mode)
        self.down1a_norm = norm_layer(128)
        self.down1b = conv_layer(128, 128, 3, padding=1, padding_mode=padding_mode)
        self.down1b_norm = norm_layer(128)
        self.down1c = pool_layer(2)
        # Down2
        self.down2a = conv_layer(128, 256, 3, padding=1, padding_mode=padding_mode)
        self.down2a_norm = norm_layer(256)
        self.down2b = conv_layer(256, 256, 3, padding=1, padding_mode=padding_mode)
        self.down2b_norm = norm_layer(256)
        # Up1
        self.up1a = conv_trans_layer(256, 128, 2, 2)
        self.up1a_norm = norm_layer(128)
        self.up1c = conv_layer(256, 128, 3, padding=1, padding_mode=padding_mode)
        self.up1c_norm = norm_layer(128)
        self.up1d = conv_layer(128, 128, 3, padding=1, padding_mode=padding_mode)
        self.up1d_norm = norm_layer(128)
        self.up1e = conv_layer(128, 128, 3, padding=1, padding_mode=padding_mode)
        self.up1e_norm = norm_layer(128)
        # Up0
        self.up0a = conv_trans_layer(128, 64, 2, 2)
        self.up0a_norm = norm_layer(64)
        self.up0c = conv_layer(128, 64, 3, padding=1, padding_mode=padding_mode)
        self.up0c_norm = norm_layer(64)
        self.up0d = conv_layer(64, 64, 3, padding=1, padding_mode=padding_mode)
        self.up0d_norm = norm_layer(64)
        self.up0e = conv_layer(64, 64, 3, padding=1, padding_mode=padding_mode)
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
    def __init__(self):
        super(DnCNN, self).__init__()
        self.num_layers = 17
        input_channels = 1
        padding = 1
        padding_mode = 'reflect'
        output_channels = 1
        inter_kernel_channel = 64
        kernel_size = 3
        self.activation_fn = F.relu
        conv_layer = nn.Conv3d
        norm_layer = nn.BatchNorm3d
        self.conv1 = conv_layer(input_channels,
                                inter_kernel_channel,
                                kernel_size,
                                padding_mode=padding_mode,
                                padding=padding)
        for num_layer in range(2, self.num_layers):
            self.__setattr__(f"conv{num_layer}", conv_layer(inter_kernel_channel,
                                                            inter_kernel_channel,
                                                            kernel_size,
                                                            padding_mode=padding_mode,
                                                            padding=padding,
                                                            bias=False))
            self.__setattr__(f"norm{num_layer}", norm_layer(inter_kernel_channel))
        self.__setattr__(f"conv{self.num_layers}", conv_layer(inter_kernel_channel,
                                                         output_channels,
                                                         kernel_size,
                                                         padding_mode=padding_mode,
                                                         padding=padding))

    def forward(self, x):
        output = self.conv1(x)
        output = self.activation_fn(output)
        output = self.conv2(output)
        output = self.norm2(output)
        output = self.activation_fn(output)
        output = self.conv3(output)
        output = self.norm3(output)
        output = self.activation_fn(output)
        output = self.conv4(output)
        output = self.norm4(output)
        output = self.activation_fn(output)
        output = self.conv5(output)
        output = self.norm5(output)
        output = self.activation_fn(output)
        output = self.conv6(output)
        output = self.norm6(output)
        output = self.activation_fn(output)
        output = self.conv7(output)
        output = self.norm7(output)
        output = self.activation_fn(output)
        output = self.conv8(output)
        output = self.norm8(output)
        output = self.activation_fn(output)
        output = self.conv9(output)
        output = self.norm9(output)
        output = self.activation_fn(output)
        output = self.conv10(output)
        output = self.norm10(output)
        output = self.activation_fn(output)
        output = self.conv11(output)
        output = self.norm11(output)
        output = self.activation_fn(output)
        output = self.conv12(output)
        output = self.norm12(output)
        output = self.activation_fn(output)
        output = self.conv13(output)
        output = self.norm13(output)
        output = self.activation_fn(output)
        output = self.conv14(output)
        output = self.norm14(output)
        output = self.activation_fn(output)
        output = self.conv15(output)
        output = self.norm15(output)
        output = self.activation_fn(output)
        output = self.conv16(output)
        output = self.norm16(output)
        output = self.activation_fn(output)
        output = self.conv17(output)
        output = x - output
        return output


class CascadedDnCNNWithUNet(nn.Module):
    def __init__(self):
        super(CascadedDnCNNWithUNet, self).__init__()
        self.dncnn0 = DnCNN()
        self.unet = UNet()

    def forward(self, x):
        # Input pre-processing
        x = x.unsqueeze_(0).unsqueeze_(0)
        low = torch.log1p(x)
        high = torch.log1p(10**7 * x)
        # Low inference
        low = self.dncnn0(low)
        low = self.unet(low)
        low = (torch.exp(low) - 1) / 10**7
        # High inference
        high = self.dncnn0(high)
        high = self.unet(high)
        high = (torch.exp(high) - 1)
        x = torch.where(x > 0.03, high, low)
        return x.squeeze()


def get_args():
    parser = argparse.ArgumentParser(description="Script for exporting weights to ONNX format")
    parser.add_argument('--checkpoint-path', type=str, help='path to the pytorch lightning checkpoint')
    parser.add_argument('--onnx-path', type=str, help='output path for the onnx checkpoint')
    return parser.parse_args()


def main():
    args = get_args()

    print("Command line arguments:")
    print(args)
    model = CascadedDnCNNWithUNet().cuda()
    model = load_model_from_lightning_checkpoint(args.checkpoint_path, model)
    scripted_model = torch.jit.script(model)
    torch.onnx.export(scripted_model, torch.rand((64, 64, 64), device='cuda'), args.onnx_path,
                      verbose=True, input_names=['noisy'], output_names=['clean'],
                      dynamic_axes={"noisy": [0, 1, 2], "clean": [0, 1, 2]})


if __name__ == '__main__':
    main()
