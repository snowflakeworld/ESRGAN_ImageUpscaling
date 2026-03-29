import tensorflow as tf
import os
from config import Config
from models import Generator, Discriminator
from losses import ESRGANLoss
from utils import create_dataset, save_image
import time


class ESRGANTrainer:
    def __init__(self, config):
        self.config = config
        self.generator = Generator(config.scale_factor, config.num_residual_blocks)
        self.discriminator = Discriminator()
        self.loss_fn = ESRGANLoss(
            config.lambda_adversarial, config.lambda_perceptual, config.lambda_pixel
        )

        self.gen_optimizer = keras.optimizers.Adam(
            config.learning_rate, beta_1=0.9, beta_2=0.999
        )
        self.disc_optimizer = keras.optimizers.Adam(
            config.learning_rate, beta_1=0.9, beta_2=0.999
        )

        self.checkpoint = tf.train.Checkpoint(
            generator=self.generator,
            discriminator=self.discriminator,
            gen_optimizer=self.gen_optimizer,
            disc_optimizer=self.disc_optimizer,
        )
        self.manager = tf.train.CheckpointManager(
            self.checkpoint, config.checkpoint_dir, max_to_keep=5
        )

        if self.manager.latest_checkpoint:
            self.checkpoint.restore(self.manager.latest_checkpoint)
            print(f"Restored from {self.manager.latest_checkpoint}")

    @tf.function
    def train_step(self, lr_images, hr_images):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate super-resolution images
            sr_images = self.generator(lr_images, training=True)

            # Discriminator outputs
            real_output = self.discriminator(hr_images, training=True)
            fake_output = self.discriminator(sr_images, training=True)

            # Calculate losses
            disc_loss = self.loss_fn.discriminator_loss(real_output, fake_output)
            gen_loss = self.loss_fn.generator_loss(hr_images, sr_images, fake_output)

        # Calculate gradients
        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_gradients = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables
        )

        # Apply gradients
        self.gen_optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables)
        )
        self.disc_optimizer.apply_gradients(
            zip(disc_gradients, self.discriminator.trainable_variables)
        )

        return gen_loss, disc_loss

    def train(self):
        train_dataset = create_dataset(
            self.config.train_lr_dir,
            self.config.train_hr_dir,
            self.config.batch_size,
            self.config.scale_factor,
        )

        for epoch in range(self.config.epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.epochs}")

            start_time = time.time()
            gen_loss_total = 0
            disc_loss_total = 0

            for step, (lr_batch, hr_batch) in enumerate(
                train_dataset.take(self.config.steps_per_epoch)
            ):
                gen_loss, disc_loss = self.train_step(lr_batch, hr_batch)
                gen_loss_total += gen_loss
                disc_loss_total += disc_loss

                if step % 100 == 0:
                    print(
                        f"Step {step}: G Loss = {gen_loss.numpy():.4f}, D Loss = {disc_loss.numpy():.4f}"
                    )

                    # Save sample
                    sample_lr = lr_batch[0:1]
                    sample_sr = self.generator(sample_lr, training=False)
                    save_image(
                        sample_sr[0].numpy(),
                        os.path.join(
                            self.config.sample_dir, f"epoch_{epoch+1}_step_{step}.png"
                        ),
                    )

            # Save checkpoint
            self.manager.save()

            end_time = time.time()
            print(f"Epoch {epoch + 1} completed in {end_time - start_time:.2f} seconds")
            print(f"Average G Loss: {gen_loss_total / self.config.steps_per_epoch:.4f}")
            print(
                f"Average D Loss: {disc_loss_total / self.config.steps_per_epoch:.4f}"
            )


if __name__ == "__main__":
    config = Config()

    # Create directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.sample_dir, exist_ok=True)

    # Train
    trainer = ESRGANTrainer(config)
    trainer.train()
