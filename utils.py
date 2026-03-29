import tensorflow as tf
import os
import numpy as np
from PIL import Image


def load_image(image_path, scale_factor=4):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Create LR image
    hr_size = tf.shape(image)[:2]
    lr_size = hr_size // scale_factor
    image_lr = tf.image.resize(image, lr_size, method="bicubic")
    image_lr = tf.image.resize(image_lr, hr_size, method="bicubic")

    return image_lr, image


def create_dataset(lr_dir, hr_dir, batch_size=16, scale_factor=4, buffer_size=1000):
    lr_paths = tf.data.Dataset.list_files(os.path.join(lr_dir, "*.png"))
    hr_paths = tf.data.Dataset.list_files(os.path.join(hr_dir, "*.png"))

    dataset = tf.data.Dataset.zip((lr_paths, hr_paths))
    dataset = dataset.map(
        lambda x, y: load_image(x, scale_factor), num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def save_image(image, path):
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(image).save(path)
