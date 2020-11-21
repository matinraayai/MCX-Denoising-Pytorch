import tensorflow as tf
import torch
import torch.nn as nn


class UNetTorch(nn.Module):
    def __init__(self):
        super(UNetTorch, self).__init__()
        self.down0a = nn.Conv2d(1, 64, 3, padding=1, padding_mode='reflect')
        self.down0a_norm = nn.BatchNorm2d(64)
        self.down0b = nn.Conv2d(64, 64, 3, padding=1, padding_mode='reflect')
        self.down0b_norm = nn.BatchNorm2d(64)
        self.down0c = nn.MaxPool2d((2, 2))
        # Down1
        self.down1a = nn.Conv2d(64, 128, 3, padding=1, padding_mode='reflect')
        self.down1a_norm = nn.BatchNorm2d(128)
        self.down1b = nn.Conv2d(128, 128, 3, padding=1, padding_mode='reflect')
        self.down1b_norm = nn.BatchNorm2d(128)
        self.down1c = nn.MaxPool2d((2, 2))
        # Down2
        self.down2a = nn.Conv2d(128, 256, 3, padding=1, padding_mode='reflect')
        self.down2a_norm = nn.BatchNorm2d(256)
        self.down2b = nn.Conv2d(256, 256, 3, padding=1, padding_mode='reflect')
        self.down2b_norm = nn.BatchNorm2d(256)
        # Up1
        self.up1a = nn.ConvTranspose2d(256, 128, 2, 2)
        self.up1a_norm = nn.BatchNorm2d(128)
        self.up1c = nn.Conv2d(256, 128, 3, padding=1, padding_mode='reflect')
        self.up1c_norm = nn.BatchNorm2d(128)
        self.up1d = nn.Conv2d(128, 128, 3, padding=1, padding_mode='reflect')
        self.up1d_norm = nn.BatchNorm2d(128)
        self.up1e = nn.Conv2d(128, 128, 3, padding=1, padding_mode='reflect')
        self.up1e_norm = nn.BatchNorm2d(128)
        # Up0
        self.up0a = nn.ConvTranspose2d(128, 64, 2, 2)
        self.up0a_norm = nn.BatchNorm2d(64)
        self.up0c = nn.Conv2d(128, 64, 3, padding=1, padding_mode='reflect')
        self.up0c_norm = nn.BatchNorm2d(64)
        self.up0d = nn.Conv2d(64, 64, 3, padding=1, padding_mode='reflect')
        self.up0d_norm = nn.BatchNorm2d(64)
        self.up0e = nn.Conv2d(64, 64, 3, padding=1, padding_mode='reflect')
        self.up0e_norm = nn.BatchNorm2d(64)
        self.last_layer = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        down0a = self.down0a_norm(self.down0a(x))
        down0b = self.down0b_norm(self.down0b(down0a))
        down0c = self.down0c(down0b)
        down1a = self.down1a_norm(self.down1a(down0c))
        down1b = self.down1b_norm(self.down1b(down1a))
        down1c = self.down1c(down1b)
        down2a = self.down2a_norm(self.down2a(down1c))
        down2b = self.down2b_norm(self.down2b(down2a))
        up1a = self.up1a_norm(self.up1a(down2b))
        up1b = torch.cat([up1a, down1b], dim=1)
        up1c = self.up1c_norm(self.up1c(up1b))
        up1d = self.up1d_norm(self.up1d(up1c))
        up1e = self.up1e_norm(self.up1e(up1d))
        up0a = self.up0a_norm(self.up0a(up1e))
        up0b = torch.cat([up0a, down0b], dim=1)
        up0c = self.up0c_norm(self.up0c(up0b))
        up0d = self.up0d_norm(self.up0d(up0c))
        up0e = self.up0e_norm(self.up0e(up0d))
        output = self.last_layer(up0e)
        return x - output


class UNet(tf.keras.Model):
    """
    UNet in TF V2
    """
    def __init__(self):
        super(UNet, self).__init__()
        # Down0
        self.down0a = tf.keras.layers.Conv2D(64, (3, 3), padding='same')
        self.down0a_norm = tf.keras.layers.BatchNormalization()
        self.down0b = tf.keras.layers.Conv2D(64, (3, 3), padding='same')
        self.down0b_norm = tf.keras.layers.BatchNormalization()
        self.down0c = tf.keras.layers.MaxPool2D((2, 2), padding='same')
        # Down1
        self.down1a = tf.keras.layers.Conv2D(128, (3, 3), padding='same')
        self.down1a_norm = tf.keras.layers.BatchNormalization()
        self.down1b = tf.keras.layers.Conv2D(128, (3, 3), padding='same')
        self.down1b_norm = tf.keras.layers.BatchNormalization()
        self.down1c = tf.keras.layers.MaxPool2D((2, 2), padding='same')
        # Down2
        self.down2a = tf.keras.layers.Conv2D(256, (3, 3), padding='same')
        self.down2a_norm = tf.keras.layers.BatchNormalization()
        self.down2b = tf.keras.layers.Conv2D(256, (3, 3), padding='same')
        self.down2b_norm = tf.keras.layers.BatchNormalization()
        # Up1
        self.up1a = tf.keras.layers.Conv2DTranspose(128, (2, 2), 2)
        self.up1a_norm = tf.keras.layers.BatchNormalization()
        self.up1c = tf.keras.layers.Conv2D(128, (3, 3), padding='same')
        self.up1c_norm = tf.keras.layers.BatchNormalization()
        self.up1d = tf.keras.layers.Conv2D(128, (3, 3), padding='same')
        self.up1d_norm = tf.keras.layers.BatchNormalization()
        self.up1e = tf.keras.layers.Conv2D(128, (3, 3), padding='same')
        self.up1e_norm = tf.keras.layers.BatchNormalization()
        # Up0
        self.up0a = tf.keras.layers.Conv2DTranspose(64, (2, 2), 2)
        self.up0a_norm = tf.keras.layers.BatchNormalization()
        self.up0c = tf.keras.layers.Conv2D(64, (3, 3), padding='same')
        self.up0c_norm = tf.keras.layers.BatchNormalization()
        self.up0d = tf.keras.layers.Conv2D(64, (3, 3), padding='same')
        self.up0d_norm = tf.keras.layers.BatchNormalization()
        self.up0e = tf.keras.layers.Conv2D(64, (3, 3), padding='same')
        self.up0e_norm = tf.keras.layers.BatchNormalization()
        self.last_layer = tf.keras.layers.Conv2D(1, (1, 1))

    def call(self, x, training=True, **kwargs):
        down0a = self.down0a_norm(self.down0a(x, training=training), training=training)
        down0b = self.down0b_norm(self.down0b(down0a, training=training), training=training)
        down0c = self.down0c(down0b, training=training)
        down1a = self.down1a_norm(self.down1a(down0c, training=training), training=training)
        down1b = self.down1b_norm(self.down1b(down1a, training=training), training=training)
        down1c = self.down1c(down1b, training=training)
        down2a = self.down2a_norm(self.down2a(down1c, training=training), training=training)
        down2b = self.down2b_norm(self.down2b(down2a, training=training), training=training)
        up1a = self.up1a_norm(self.up1a(down2b, training=training), training=training)
        up1b = tf.keras.layers.concatenate([up1a, down1b], axis=3)
        print(up1b.shape)
        up1c = self.up1c_norm(self.up1c(up1b, training=training), training=training)
        up1d = self.up1d_norm(self.up1d(up1c, training=training), training=training)
        up1e = self.up1e_norm(self.up1e(up1d, training=training), training=training)
        up0a = self.up0a_norm(self.up0a(up1e, training=training), training=training)
        up0b = tf.keras.layers.concatenate([up0a, down0b], axis=3)
        up0c = self.up0c_norm(self.up0c(up0b, training=training), training=training)
        up0d = self.up0d_norm(self.up0d(up0c, training=training), training=training)
        up0e = self.up0e_norm(self.up0e(up0d, training=training), training=training)
        output = self.last_layer(up0e, training=training)
        return x - output


class Selu(tf.keras.Model):
    def __init__(self, scale=1.0507009873554804934193349852946, alpha=1.6732632423543772848170429916717):
        super(Selu, self).__init__()
        self.scale = scale
        self.alpha = alpha

    def __call__(self, x):
        return self.scale * tf.where(x >= 0.0, x, self.alpha * tf.nn.elu(x))


class DnCNN(tf.keras.Model):
    def __init__(self, output_channels=1, num_layers=17, activation_fn=tf.nn.relu):
        super(DnCNN, self).__init__()
        self.num_layers = num_layers
        self.activation_fn = activation_fn
        self.__setattr__("conv1", tf.keras.layers.Conv2D(64, 3, padding='same', activation=activation_fn))
        for num_layer in range(2, num_layers):
            self.__setattr__(f"conv{num_layer}", tf.keras.layers.Conv2D(64, 3, padding='same', use_bias=False))
            self.__setattr__(f"norm{num_layer}", tf.keras.layers.BatchNormalization())
        self.__setattr__(f"conv{num_layers}", tf.keras.layers.Conv2D(output_channels, 3, padding='same'))

    def __call__(self, x, training=True):
        output = self.conv1(x, training=True)
        for num_layer in range(2, self.num_layers):
            output = self.__getattribute__(f"conv{num_layer}")(output, training=training)
            output = self.__getattribute__(f"norm{num_layer}")(output, training=training)
            output = self.activation_fn(output)
        output = self.__getattribute__(f"conv{self.num_layers}")(output, training=training)
        return x - output


class DnCNNWithUNet(tf.keras.Model):
    def __init__(self, output_channels=1, num_dcnn_layers=17, activation_fn=tf.nn.relu):
        super(DnCNNWithUNet, self).__init__()
        self.dncnn = DnCNN(output_channels=output_channels, num_layers=num_dcnn_layers, activation_fn=activation_fn)
        self.unet = UNet()

    def __call__(self, x, training=True):
        output = self.dncnn(x, training=training)
        return self.unet(output, training=training)


class CascadedDnCNNWithUNet(tf.keras.Model):
    def __init__(self, num_dcnn=3, output_channels=1, num_dcnn_layers=17, activation_fn=tf.compat.v1.nn.relu):
        super(CascadedDnCNNWithUNet, self).__init__()
        self.num_dcnn = num_dcnn
        for num in range(self.num_dcnn):
            self.__setattr__(f"dncnn{num}", DnCNN(output_channels=output_channels,
                                                  num_layers=num_dcnn_layers, activation_fn=activation_fn))
        self.unet = UNet()

    def __call__(self, x, training=True):
        for num in range(self.num_dcnn):
            x = self.__getattribute__(f"dncnn{num}")(x, training=training)
        return self.unet(x, training=training)
