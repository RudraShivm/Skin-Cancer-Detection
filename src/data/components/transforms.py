"""Data augmentation transforms for ISIC 2024 3D-TBP images.

ISIC 2024 images are crops from 3D Total Body Photography (3D-TBP). They often
contain black borders, hair artifacts, and varying illumination. The augmentations
here are designed to handle these domain-specific characteristics.

Reference: Yakiniku 2nd place solution used aggressive augmentations including
hair simulation, microscope-style borders, and heavy cropping.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(img_size: int = 224):
    """Get training transforms with domain-specific augmentations for 3D-TBP images.

    Key augmentations:
    - RandomResizedCrop: Forces model to learn from different lesion regions and
      handles black borders by zooming into relevant areas.
    - Transpose + Flips + Rotate90: Full geometric invariance for lesions
      (orientation doesn't matter for pathology).
    - HueSaturationValue + RandomBrightnessContrast: Handles color variation
      across different 3D-TBP devices and lighting conditions.
    - CoarseDropout: Simulates hair and artifact occlusion (like AdvancedHairAug
      used in winning solutions) by dropping rectangular patches.
    """
    return A.Compose([
        A.RandomResizedCrop(
            height=img_size,
            width=img_size,
            scale=(0.7, 1.0),     # Crop 70-100% of the image
            ratio=(0.75, 1.3333), # Slight aspect ratio variation
        ),
        A.Transpose(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=30,     # Broader rotation than before (was 15)
            border_mode=0,       # Black border fill (matches 3D-TBP edges)
            p=0.5,
        ),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
        ], p=0.3),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=20,
            p=0.5,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5,
        ),
        A.CoarseDropout(
            max_holes=8,
            max_height=img_size // 12,
            max_width=img_size // 12,
            min_holes=1,
            fill_value=0,        # Black fill simulates hair/artifacts
            p=0.3,
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_val_transforms(img_size: int = 224):
    """Get validation/test transforms (no augmentation, deterministic)."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])