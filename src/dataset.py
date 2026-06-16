"""
PyTorch Dataset for Multi-Magnification Colposcopy.

Loads 4 magnification images per patient (either pre-acid or post-acid),
applies consistent spatial augmentations across all views, and returns
a stacked tensor ready for the attention MIL model.

Augmentation is fully configurable via the ``augmentation`` block in
the YAML config. Supported operations: RandomResizedCrop, ColorJitter,
GaussianBlur, RandomErasing, RandomAffine. Falls back to sane defaults
if keys are absent.

Usage:
    dataset = ColposcopyMagnificationDataset(manifest_df, condition="post", mode="train", cfg=cfg)
"""

import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFilter
from torchvision import transforms


class ColposcopyMagnificationDataset(Dataset):
    """
    Multi-magnification colposcopy dataset.

    Each sample returns 4 images (one per magnification level) for a single
    patient, along with the binary label.

    Args:
        manifest_df: DataFrame from manifeste.csv
        condition: "pre" or "post" — which acid condition to use
        mode: "train" or "eval"
        cfg: Configuration dict
        magnifications: List of magnification indices to include (0-3).
                       None = all 4. Used for ablation studies.
    """

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        condition: str = "post",
        mode: str = "train",
        cfg: dict = None,
        magnifications: list = None,
    ):
        self.df = manifest_df.reset_index(drop=True)
        self.condition = condition
        self.mode = mode
        self.cfg = cfg
        self.image_size = cfg["data"]["image_size"]

        # Augmentation config (V2 — with backward-compat defaults)
        self.aug_cfg = cfg.get("augmentation", {})

        # Determine which magnifications to use
        if magnifications is not None:
            self.mag_indices = magnifications
        else:
            self.mag_indices = list(range(cfg["data"]["num_magnifications"]))

        # Build column names for the image paths
        if condition == "both":
            # Dual-stream: load all pre AND post images (8 total for 4 magnifications)
            self.path_columns = (
                [f"pre_{i+1}_path" for i in self.mag_indices] +
                [f"post_{i+1}_path" for i in self.mag_indices]
            )
        else:
            prefix = "pre" if condition == "pre" else "post"
            self.path_columns = [f"{prefix}_{i+1}_path" for i in self.mag_indices]

        # Build transforms
        self.spatial_transforms = self._build_spatial_transforms()
        self.color_transform = self._build_color_transform()
        self.post_tensor_transform = self._build_post_tensor_transform()
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _build_spatial_transforms(self):
        """Build spatial transforms (applied consistently across magnifications)."""
        if self.mode == "train":
            crop_pad = self.aug_cfg.get("random_crop_padding", 32)
            rotation = self.aug_cfg.get("random_rotation", 15)
            h_flip = self.aug_cfg.get("random_horizontal_flip", True)
            v_flip = self.aug_cfg.get("random_vertical_flip", True)

            transform_list = []
            
            crop_scale = self.aug_cfg.get("random_crop_scale", None)
            if crop_scale is not None:
                # RandomResizedCrop scale must be <= 1.0 in PyTorch (it's the area proportion)
                s_min = max(0.1, min(1.0, float(crop_scale[0])))
                s_max = max(s_min, min(1.0, float(crop_scale[1])))
                transform_list.append(
                    transforms.RandomResizedCrop(
                        size=self.image_size,
                        scale=(s_min, s_max),
                        ratio=(0.9, 1.1) # Maintain roughly square aspect ratio
                    )
                )
            else:
                crop_pad = self.aug_cfg.get("random_crop_padding", 32)
                transform_list.extend([
                    transforms.Resize((self.image_size + crop_pad, self.image_size + crop_pad)),
                    transforms.RandomCrop(self.image_size),
                ])

            if h_flip:
                transform_list.append(transforms.RandomHorizontalFlip())
            if v_flip:
                transform_list.append(transforms.RandomVerticalFlip())

            transform_list.append(transforms.RandomRotation(rotation))

            # V2: Random Affine (scale + shear)
            affine_cfg = self.aug_cfg.get("random_affine", {})
            if affine_cfg.get("enabled", False):
                transform_list.append(transforms.RandomAffine(
                    degrees=affine_cfg.get("degrees", 0),
                    scale=tuple(affine_cfg.get("scale", [0.9, 1.1])),
                    shear=affine_cfg.get("shear", 5),
                ))

            return transforms.Compose(transform_list)
        else:
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
            ])

    def _build_color_transform(self):
        """Build color augmentation (applied independently per image)."""
        if self.mode == "train":
            # Read from augmentation config or use defaults
            cj = self.aug_cfg.get("color_jitter", {})
            brightness = cj.get("brightness", 0.2)
            contrast = cj.get("contrast", 0.2)
            saturation = cj.get("saturation", 0.1)
            hue = cj.get("hue", 0.05)

            transform_list = [
                transforms.ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=hue,
                ),
            ]

            # V2: Gaussian blur with probability
            blur_cfg = self.aug_cfg.get("gaussian_blur", {})
            if blur_cfg.get("enabled", False):
                prob = blur_cfg.get("probability", 0.2)
                ks = blur_cfg.get("kernel_size", 5)
                transform_list.append(
                    transforms.RandomApply(
                        [transforms.GaussianBlur(kernel_size=ks)],
                        p=prob,
                    )
                )

            return transforms.Compose(transform_list)
        else:
            return None

    def _build_post_tensor_transform(self):
        """Build transforms applied AFTER ToTensor (e.g. Random Erasing)."""
        if self.mode == "train":
            erase_cfg = self.aug_cfg.get("random_erasing", {})
            if erase_cfg.get("enabled", False):
                prob = erase_cfg.get("probability", 0.15)
                scale = tuple(erase_cfg.get("scale", [0.02, 0.15]))
                return transforms.RandomErasing(p=prob, scale=scale)
        return None

    @property
    def effective_mag_labels(self):
        """Return human-readable labels for each image slot (used for attention weight columns)."""
        raw = ["1x", "3x", "5x", "7x"]
        if self.condition == "both":
            return (
                [f"pre_{raw[i]}" for i in self.mag_indices] +
                [f"post_{raw[i]}" for i in self.mag_indices]
            )
        else:
            return [raw[i] if i < len(raw) else f"mag{i}" for i in self.mag_indices]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = torch.tensor(row["label"], dtype=torch.float32)
        patient_id = row["patient_id"]

        # Load all magnification images
        images = []
        for col in self.path_columns:
            img_path = row[col]
            img = Image.open(img_path).convert("RGB")
            images.append(img)

        # Apply CONSISTENT spatial transforms across all magnifications
        # We do this by setting the same random seed before each spatial transform
        if self.mode == "train":
            spatial_seed = random.randint(0, 2**32 - 1)

        processed_images = []
        for img in images:
            # Consistent spatial transforms
            if self.mode == "train":
                torch.manual_seed(spatial_seed)
                random.seed(spatial_seed)
                np.random.seed(spatial_seed % (2**31))

            img = self.spatial_transforms(img)

            # Independent color augmentation per image
            if self.color_transform is not None:
                img = self.color_transform(img)

            # Normalize and convert to tensor
            img = self.normalize(img)

            # Post-tensor augmentation (Random Erasing)
            if self.post_tensor_transform is not None:
                img = self.post_tensor_transform(img)

            processed_images.append(img)

        # Stack: [num_magnifications, 3, H, W]
        images_tensor = torch.stack(processed_images, dim=0)

        return images_tensor, label, patient_id
