import tensorflow as tf
import os
import numpy as np
from PIL import Image
from config import Config
from models import Generator


class ESRGANTester:
    def __init__(self, config, checkpoint_path=None):
        self.config = config
        self.generator = Generator(config.scale_factor, config.num_residual_blocks)

        if checkpoint_path:
            self.generator.load_weights(checkpoint_path)
            print(f"Loaded weights from {checkpoint_path}")

    def test_single_image(self, image_path, output_path):
        # Load image
        image = np.array(Image.open(image_path).convert("RGB"))
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, 0)

        # Generate super-resolution
        sr_image = self.generator(image, training=False)
        sr_image = sr_image[0].numpy()

        # Save result
        sr_image = np.clip(sr_image * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(sr_image).save(output_path)
        print(f"Saved super-resolution image to {output_path}")

    def test_directory(self, input_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        for filename in os.listdir(input_dir):
            if filename.endswith((".png", ".jpg", ".jpeg")):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, f"sr_{filename}")
                self.test_single_image(input_path, output_path)

        print(f"Processed all images in {input_dir}")


if __name__ == "__main__":
    config = Config()

    # Load latest checkpoint
    checkpoint_dir = tf.train.latest_checkpoint(config.checkpoint_dir)

    # Test
    tester = ESRGANTester(config, checkpoint_dir)

    # Test single image
    # tester.test_single_image('test_input.png', 'test_output.png')

    # Test directory
    tester.test_directory(config.test_dir, "test_results")
