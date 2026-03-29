import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K


class VGGFeatureExtractor(keras.Model):
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        vgg = keras.applications.VGG19(include_top=False, weights="imagenet")
        vgg.trainable = False
        self.vgg = keras.Model(
            inputs=vgg.input, outputs=vgg.get_layer("block5_conv4").output
        )

    def call(self, x):
        x = tf.keras.applications.vgg19.preprocess_input(x * 255.0)
        return self.vgg(x)


class ESRGANLoss:
    def __init__(
        self, lambda_adversarial=5e-3, lambda_perceptual=1e-2, lambda_pixel=1e-3
    ):
        self.lambda_adv = lambda_adversarial
        self.lambda_per = lambda_perceptual
        self.lambda_pix = lambda_pixel
        self.vgg = VGGFeatureExtractor()
        self.bce = keras.losses.BinaryCrossentropy()

    def perceptual_loss(self, y_true, y_pred):
        true_features = self.vgg(y_true)
        pred_features = self.vgg(y_pred)
        return K.mean(K.square(true_features - pred_features))

    def pixel_loss(self, y_true, y_pred):
        return K.mean(K.abs(y_true - y_pred))

    def adversarial_loss(self, y_true, y_pred):
        return self.bce(y_true, y_pred)

    def generator_loss(self, y_true, y_pred, discriminator_output):
        per_loss = self.perceptual_loss(y_true, y_pred)
        pix_loss = self.pixel_loss(y_true, y_pred)
        adv_loss = self.adversarial_loss(
            tf.ones_like(discriminator_output), discriminator_output
        )

        total_loss = (
            self.lambda_per * per_loss
            + self.lambda_pix * pix_loss
            + self.lambda_adv * adv_loss
        )
        return total_loss

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.adversarial_loss(tf.ones_like(real_output), real_output)
        fake_loss = self.adversarial_loss(tf.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss
