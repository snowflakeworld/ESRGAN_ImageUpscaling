import os
import cv2
import numpy as np
from pathlib import Path


def prepare_data(hr_dir, lr_dir, scale=4):
    os.makedirs(lr_dir, exist_ok=True)

    hr_images = list(Path(hr_dir).glob("*.png")) + list(Path(hr_dir).glob("*.jpg"))

    print(f"Found {len(hr_images)} images. Generating LR images...")

    for img_path in hr_images:
        # Read HR image
        img_hr = cv2.imread(str(img_path))
        if img_hr is None:
            continue

        # Calculate LR dimensions
        h, w = img_hr.shape[:2]
        lr_h, lr_w = h // scale, w // scale

        # Downscale using Bicubic (Standard for ESRGAN)
        img_lr = cv2.resize(img_hr, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)

        # Save LR image
        save_path = os.path.join(lr_dir, img_path.name)
        cv2.imwrite(save_path, img_lr)

    print("Done!")


if __name__ == "__main__":
    # Adjust paths to where you downloaded DIV2K
    HR_PATH = "data/DIV2K_train_HR"
    LR_PATH = "data/DIV2K_train_LR"

    prepare_data(HR_PATH, LR_PATH, scale=4)
