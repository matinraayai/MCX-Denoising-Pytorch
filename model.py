import tensorflow as tf


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
        self.up1c = tf.keras.layers.Conv2D(128, (3, 3))
        self.up1c_norm = tf.keras.layers.BatchNormalization()
        self.up1d = tf.keras.layers.Conv2D(128, (3, 3))
        self.up1d_norm = tf.keras.layers.BatchNormalization()
        self.up1e = tf.keras.layers.Conv2D(128, (3, 3))
        self.up1e_norm = tf.keras.layers.BatchNormalization()
        # Up0
        self.up0a = tf.keras.layers.Conv2DTranspose(64, (2, 2), 2)
        self.up0a_norm = tf.keras.layers.BatchNormalization()
        self.up0c = tf.keras.layers.Conv2D(64, (3, 3))
        self.up0c_norm = tf.keras.layers.BatchNormalization()
        self.up0d = tf.keras.layers.Conv2D(64, (3, 3))
        self.up0d_norm = tf.keras.layers.BatchNormalization()
        self.up0e = tf.keras.layers.Conv2D(64, (3, 3))
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
        up1b = tf.keras.layers.concatenate([up1a, down1b], axis=3, training=training)
        up1c = self.up1c_norm(self.up1c(up1b, training=training), training=training)
        up1d = self.up1d_norm(self.up1d(up1c, training=training), training=training)
        up1e = self.up1e_norm(self.up1e(up1d, training=training), training=training)
        up0a = self.up0a_norm(self.up0a(up1e, training=training), training=training)
        up0b = tf.keras.layers.concatenate([up0a, down0b], axis=3, training=training)
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
