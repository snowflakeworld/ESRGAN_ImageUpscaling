import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class ResidualDenseBlock(layers.Layer):
    def __init__(self, num_filters=64, kernel_size=3):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = layers.Conv2D(
            num_filters,
            kernel_size,
            padding="same",
            activation="leaky_relu",
            kernel_initializer="he_normal",
        )
        self.conv2 = layers.Conv2D(
            num_filters,
            kernel_size,
            padding="same",
            activation="leaky_relu",
            kernel_initializer="he_normal",
        )
        self.conv3 = layers.Conv2D(
            num_filters,
            kernel_size,
            padding="same",
            activation="leaky_relu",
            kernel_initializer="he_normal",
        )
        self.conv4 = layers.Conv2D(
            num_filters,
            kernel_size,
            padding="same",
            activation="leaky_relu",
            kernel_initializer="he_normal",
        )
        self.conv5 = layers.Conv2D(
            num_filters, kernel_size, padding="same", kernel_initializer="he_normal"
        )
        self.alpha = 0.2

    def call(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x + x1)
        x3 = self.conv3(x + x1 + x2)
        x4 = self.conv4(x + x1 + x2 + x3)
        x5 = self.conv5(x + x1 + x2 + x3 + x4)
        return x + self.alpha * x5


class ResidualInResidualDenseBlock(layers.Layer):
    def __init__(self, num_filters=64, num_residual_blocks=3):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.rdb_blocks = [
            ResidualDenseBlock(num_filters) for _ in range(num_residual_blocks)
        ]
        self.alpha = 0.2

    def call(self, x):
        out = x
        for rdb in self.rdb_blocks:
            out = rdb(out)
        return x + self.alpha * out


class Generator(keras.Model):
    def __init__(self, scale_factor=4, num_residual_blocks=23):
        super(Generator, self).__init__()

        # Initial convolution
        self.conv1 = layers.Conv2D(
            64,
            3,
            padding="same",
            activation="leaky_relu",
            kernel_initializer="he_normal",
        )

        # Residual in Residual Dense Blocks
        self.rrdb_blocks = keras.Sequential(
            [ResidualInResidualDenseBlock(64, 3) for _ in range(num_residual_blocks)]
        )

        # Post convolution
        self.conv2 = layers.Conv2D(
            64, 3, padding="same", kernel_initializer="he_normal"
        )

        # Upsampling
        self.upsample1 = self._upsample_block(64, 256, scale_factor // 2)
        self.upsample2 = self._upsample_block(64, 256, scale_factor // 2)

        # Output
        self.output_conv = layers.Conv2D(3, 3, padding="same", activation="tanh")

    def _upsample_block(self, input_filters, output_filters, scale):
        return keras.Sequential(
            [
                layers.Conv2D(
                    output_filters, 3, padding="same", activation="leaky_relu"
                ),
                layers.Lambda(lambda x: tf.nn.depth_to_space(x, scale)),
            ]
        )

    def call(self, x):
        x = tf.keras.applications.vgg19.preprocess_input(x * 255.0) / 255.0
        x1 = self.conv1(x)
        x2 = self.rrdb_blocks(x1)
        x2 = self.conv2(x2)
        x = x1 + x2
        x = self.upsample1(x)
        x = self.upsample2(x)
        return self.output_conv(x)


class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = layers.Conv2D(64, 3, padding="same", activation="leaky_relu")
        self.conv2 = layers.Conv2D(64, 4, 2, padding="same", activation="leaky_relu")

        self.conv3 = layers.Conv2D(128, 3, padding="same", activation="leaky_relu")
        self.conv4 = layers.Conv2D(128, 4, 2, padding="same", activation="leaky_relu")

        self.conv5 = layers.Conv2D(256, 3, padding="same", activation="leaky_relu")
        self.conv6 = layers.Conv2D(256, 4, 2, padding="same", activation="leaky_relu")

        self.conv7 = layers.Conv2D(512, 3, padding="same", activation="leaky_relu")
        self.conv8 = layers.Conv2D(512, 4, 2, padding="same", activation="leaky_relu")

        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(100, activation="leaky_relu")
        self.dense2 = layers.Dense(1, activation="sigmoid")

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)
