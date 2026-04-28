# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from Detectron2 by Facebook, Inc. and its affiliates,
# used under the Apache 2.0 License.
# ---------------------------------------------------------------

import torch
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F
from torchvision.tv_tensors import wrap, TVTensor
from torch import nn, Tensor
from typing import Any, Union


class Transforms(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        color_jitter_enabled: bool,
        scale_range: tuple[float, float],
        max_brightness_delta: int = 32,
        max_contrast_factor: float = 0.5,
        saturation_factor: float = 0.5,
        max_hue_delta: int = 18,
        sensor_noise_enabled: bool = True,
        blur_enabled: bool = True,
        blur_kernel_size: tuple[int, int] = (3, 7),
        noise_variance: float = 0.05,
    ):
        super().__init__()

        self.img_size = img_size
        self.color_jitter_enabled = color_jitter_enabled
        self.max_brightness_factor = max_brightness_delta / 255.0
        self.max_contrast_factor = max_contrast_factor
        self.max_saturation_factor = saturation_factor
        self.max_hue_delta = max_hue_delta / 360.0

        self.random_horizontal_flip = T.RandomHorizontalFlip()
        self.scale_jitter = T.ScaleJitter(target_size=img_size, scale_range=scale_range)
        self.random_crop = T.RandomCrop(img_size)

        self.sensor_noise_enabled = sensor_noise_enabled
        self.blur_enabled = blur_enabled
        self.noise_variance = noise_variance
        
        self.gaussian_blur = T.GaussianBlur(kernel_size=blur_kernel_size, sigma=(0.1, 2.0))

    def _random_factor(self, factor: float, center: float = 1.0):
        return torch.empty(1).uniform_(center - factor, center + factor).item()

    def _brightness(self, img):
        if torch.rand(()) < 0.5:
            img = F.adjust_brightness(
                img, self._random_factor(self.max_brightness_factor)
            )

        return img

    def _contrast(self, img):
        if torch.rand(()) < 0.5:
            img = F.adjust_contrast(img, self._random_factor(self.max_contrast_factor))

        return img

    def _saturation_and_hue(self, img):
        if torch.rand(()) < 0.5:
            img = F.adjust_saturation(
                img, self._random_factor(self.max_saturation_factor)
            )

        if torch.rand(()) < 0.5:
            img = F.adjust_hue(img, self._random_factor(self.max_hue_delta, center=0.0))

        return img

    def color_jitter(self, img):
        if not self.color_jitter_enabled:
            return img

        img = self._brightness(img)

        if torch.rand(()) < 0.5:
            img = self._contrast(img)
            img = self._saturation_and_hue(img)
        else:
            img = self._saturation_and_hue(img)
            img = self._contrast(img)

        return img

    def add_sensor_noise(self, img: Tensor) -> Tensor:
        if not self.sensor_noise_enabled:
            return img

        if torch.rand(()) < 0.5:
            # Add Gaussian noise approximating sensor ISO grain
            variance = torch.rand(()) * self.noise_variance
            # Ensure noise scaling matches image tensor scale (usually 0-255 uint8)
            noise = torch.randn_like(img, dtype=torch.float32) * (variance * 255.0)
            img = img.to(torch.float32) + noise
            img = torch.clamp(img, 0, 255).to(torch.uint8)

        return img

    def pad(
        self, img: Tensor, target: dict[str, Any]
    ) -> tuple[Tensor, dict[str, Union[Tensor, TVTensor]]]:
        pad_h = max(0, self.img_size[-2] - img.shape[-2])
        pad_w = max(0, self.img_size[-1] - img.shape[-1])
        padding = [0, 0, pad_w, pad_h]

        img = F.pad(img, padding)
        target["masks"] = F.pad(target["masks"], padding)

        return img, target

    def _filter(self, target: dict[str, Union[Tensor, TVTensor]], keep: Tensor) -> dict:
        return {k: wrap(v[keep], like=v) for k, v in target.items()}

    def forward(
        self, img: Tensor, target: dict[str, Union[Tensor, TVTensor]]
    ) -> tuple[Tensor, dict[str, Union[Tensor, TVTensor]]]:
        img_orig, target_orig = img, target

        target = self._filter(target, ~target["is_crowd"])

        img = self.color_jitter(img)
        img, target = self.random_horizontal_flip(img, target)
        img, target = self.scale_jitter(img, target)
        img, target = self.pad(img, target)
        img, target = self.random_crop(img, target)

        if self.blur_enabled and torch.rand(()) < 0.25:
            img = self.gaussian_blur(img)
            
        img = self.add_sensor_noise(img)

        valid = target["masks"].flatten(1).any(1)
        if not valid.any():
            return self(img_orig, target_orig)

        target = self._filter(target, valid)

        return img, target
