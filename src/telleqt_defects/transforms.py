from __future__ import annotations

from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights


def build_transforms(image_size: int, train: bool):
    weights = EfficientNet_B0_Weights.DEFAULT
    mean = weights.transforms().mean
    std = weights.transforms().std

    if train:
        return transforms.Compose(
            [
                transforms.Resize(int(image_size * 1.12)),
                transforms.RandomResizedCrop(image_size, scale=(0.82, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.08, hue=0.02)],
                    p=0.5,
                ),
                transforms.RandomRotation(degrees=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.12)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
