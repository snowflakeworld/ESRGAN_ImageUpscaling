import tensorflow as tf


class Config:
    # Data
    train_lr_dir = "data/DIV2K_train_HR"
    train_hr_dir = "data/DIV2K_train_LR"
    test_dir = "data/test"

    # Model
    scale_factor = 4
    num_residual_blocks = 23

    # Training
    batch_size = 16
    learning_rate = 1e-4
    epochs = 100
    steps_per_epoch = 1000

    # Loss weights
    lambda_adversarial = 5e-3
    lambda_perceptual = 1e-2
    lambda_pixel = 1e-3

    # Checkpoints
    checkpoint_dir = "checkpoints"
    sample_dir = "samples"

    # Hardware
    gpu_memory_fraction = 0.9
