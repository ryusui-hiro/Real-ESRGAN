"""Training utilities for EfficientSR: datasets and loss functions."""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def _load_image(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    array = np.array(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return tensor


def _random_crop(hr: torch.Tensor, patch_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    _, h, w = hr.shape
    if h < patch_size or w < patch_size:
        raise ValueError(
            f"HR patch size {patch_size} exceeds image dimensions {(h, w)}."
        )
    top = random.randint(0, h - patch_size)
    left = random.randint(0, w - patch_size)
    return hr[:, top : top + patch_size, left : left + patch_size], (top, left)


@dataclass
class DataloaderConfig:
    hr_dir: Path
    lr_dir: Optional[Path] = None
    batch_size: int = 4
    num_workers: int = 0
    scale: int = 4
    patch_size: Optional[int] = 128
    augment: bool = True
    shuffle: bool = True


class SuperResolutionDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Dataset for paired LR/HR images with optional on-the-fly downsampling.

    If ``lr_dir`` is not provided, LR images are created by bicubic resizing the
    HR image with the configured ``scale``. When both HR and LR directories are
    provided, files are matched by filename.
    """

    def __init__(self, hr_dir: Path, lr_dir: Optional[Path] = None, scale: int = 4,
                 patch_size: Optional[int] = 128, augment: bool = True) -> None:
        super().__init__()
        if patch_size is not None and patch_size % scale != 0:
            raise ValueError("patch_size must be divisible by scale for aligned crops.")
        self.hr_dir = Path(hr_dir)
        self.lr_dir = Path(lr_dir) if lr_dir is not None else None
        self.scale = scale
        self.patch_size = patch_size
        self.augment = augment
        self.hr_files = self._gather_files(self.hr_dir)

    @staticmethod
    def _gather_files(folder: Path) -> List[Path]:
        if not folder.exists():
            raise FileNotFoundError(f"Directory not found: {folder}")
        files: List[Path] = [p for p in folder.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}]
        if not files:
            raise FileNotFoundError(f"No image files found in {folder}")
        return sorted(files)

    def __len__(self) -> int:
        return len(self.hr_files)

    def _paired_lr_path(self, hr_path: Path) -> Optional[Path]:
        if self.lr_dir is None:
            return None
        candidate = self.lr_dir / hr_path.name
        return candidate if candidate.exists() else None

    def _load_hr_lr(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        hr_path = self.hr_files[index]
        hr = _load_image(hr_path)

        lr_path = self._paired_lr_path(hr_path)
        if lr_path is None:
            h, w = hr.shape[1:]
            lr_size = (w // self.scale, h // self.scale)
            lr_image = Image.fromarray((hr.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8))
            lr_image = lr_image.resize(lr_size, resample=Image.Resampling.BICUBIC)
            lr = torch.from_numpy(np.array(lr_image, dtype=np.float32) / 255.0).permute(2, 0, 1)
        else:
            lr = _load_image(lr_path)
        return hr, lr

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        hr, lr = self._load_hr_lr(index)

        if self.patch_size is not None:
            hr, (top, left) = _random_crop(hr, self.patch_size)
            lr_patch = self.patch_size // self.scale
            lr = lr[:, top // self.scale : top // self.scale + lr_patch, left // self.scale : left // self.scale + lr_patch]

        if self.augment:
            if random.random() < 0.5:
                hr = torch.flip(hr, dims=[2])
                lr = torch.flip(lr, dims=[2])
            if random.random() < 0.5:
                hr = torch.flip(hr, dims=[1])
                lr = torch.flip(lr, dims=[1])

        return lr.contiguous(), hr.contiguous()


class CharbonnierLoss(nn.Module):
    """Charbonnier loss for robust pixel differences."""

    def __init__(self, epsilon: float = 1e-3) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.epsilon**2))


@dataclass
class LossConfig:
    pixel_weight: float = 1.0
    charbonnier_weight: float = 0.0
    tv_weight: float = 0.0
    charbonnier_epsilon: float = 1e-3


class CompositeLoss(nn.Module):
    """Combine multiple SR losses with configurable weights."""

    def __init__(self, config: LossConfig) -> None:
        super().__init__()
        self.config = config
        self.l1 = nn.L1Loss()
        self.charbonnier = CharbonnierLoss(epsilon=config.charbonnier_epsilon)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        components: Dict[str, torch.Tensor] = {}
        total = torch.zeros(1, device=pred.device, dtype=pred.dtype)

        if self.config.pixel_weight > 0:
            l1 = self.l1(pred, target)
            components["l1"] = l1
            total = total + self.config.pixel_weight * l1

        if self.config.charbonnier_weight > 0:
            char = self.charbonnier(pred, target)
            components["charbonnier"] = char
            total = total + self.config.charbonnier_weight * char

        if self.config.tv_weight > 0:
            tv = self._total_variation(pred)
            components["tv"] = tv
            total = total + self.config.tv_weight * tv

        return total, components

    @staticmethod
    def _total_variation(x: torch.Tensor) -> torch.Tensor:
        dh = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
        dw = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
        return dh + dw


def build_dataloaders(train_config: DataloaderConfig, val_config: Optional[DataloaderConfig] = None) -> Dict[str, DataLoader]:
    train_dataset = SuperResolutionDataset(
        hr_dir=train_config.hr_dir,
        lr_dir=train_config.lr_dir,
        scale=train_config.scale,
        patch_size=train_config.patch_size,
        augment=train_config.augment,
    )

    loaders: Dict[str, DataLoader] = {
        "train": DataLoader(
            train_dataset,
            batch_size=train_config.batch_size,
            shuffle=train_config.shuffle,
            num_workers=train_config.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    }

    if val_config is not None:
        val_dataset = SuperResolutionDataset(
            hr_dir=val_config.hr_dir,
            lr_dir=val_config.lr_dir,
            scale=val_config.scale,
            patch_size=val_config.patch_size,
            augment=False,
        )
        loaders["val"] = DataLoader(
            val_dataset,
            batch_size=val_config.batch_size,
            shuffle=False,
            num_workers=val_config.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    return loaders


def build_loss(config: Optional[LossConfig] = None) -> CompositeLoss:
    return CompositeLoss(config or LossConfig())
