"""Training utilities for EfficientSR: datasets and loss functions."""
from __future__ import annotations

import random
import shutil
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


DIV2K_BASE_URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K/"


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


def _div2k_split_folders(split: str, subset: str, scale: int) -> Tuple[str, str, str, str]:
    split_name = "train" if split == "train" else "valid"
    hr_dirname = f"DIV2K_{split_name}_HR"
    lr_dirname = f"DIV2K_{split_name}_LR_{subset}_X{scale}"
    hr_zip = f"{hr_dirname}.zip"
    lr_zip = f"DIV2K_{split_name}_LR_{subset}_X{scale}.zip"
    return hr_dirname, lr_dirname, hr_zip, lr_zip


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, dest.open("wb") as f:
        shutil.copyfileobj(response, f)


def _extract_zip(archive: Path, target_dir: Path) -> None:
    with zipfile.ZipFile(archive, "r") as zip_ref:
        zip_ref.extractall(target_dir)


def _download_div2k_split(
    root: Path,
    split: str,
    subset: str,
    scale: int,
    source: str,
    hf_repo: str,
) -> Tuple[Path, Path]:
    hr_dirname, lr_dirname, hr_zip, lr_zip = _div2k_split_folders(split, subset, scale)
    hr_dir = root / hr_dirname
    lr_dir = root / lr_dirname

    if hr_dir.exists() and lr_dir.exists():
        return hr_dir, lr_dir

    root.mkdir(parents=True, exist_ok=True)

    if source == "huggingface":
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise ImportError(
                "huggingface_hub is required for huggingface downloads. Install it or set download_source='official'."
            ) from exc

        hr_archive = Path(
            hf_hub_download(
                repo_id=hf_repo,
                filename=hr_zip,
                repo_type="dataset",
            )
        )
        lr_archive = Path(
            hf_hub_download(
                repo_id=hf_repo,
                filename=lr_zip,
                repo_type="dataset",
            )
        )
    else:
        hr_archive = root / hr_zip
        lr_archive = root / lr_zip
        if not hr_archive.exists():
            _download_file(DIV2K_BASE_URL + hr_zip, hr_archive)
        if not lr_archive.exists():
            _download_file(DIV2K_BASE_URL + lr_zip, lr_archive)

    _extract_zip(hr_archive, root)
    _extract_zip(lr_archive, root)
    return hr_dir, lr_dir


def prepare_div2k_folders(config: Div2KConfig, split: str = "train") -> Tuple[Path, Path]:
    hr_dirname, lr_dirname, *_ = _div2k_split_folders(split, config.subset, config.scale)
    hr_dir = config.root / hr_dirname
    lr_dir = config.root / lr_dirname

    if not hr_dir.exists() or not lr_dir.exists():
        if not config.download:
            raise FileNotFoundError(
                f"DIV2K split '{split}' was not found under {config.root}. "
                "Enable download=True to fetch it automatically."
            )
        hr_dir, lr_dir = _download_div2k_split(
            root=config.root,
            split=split,
            subset=config.subset,
            scale=config.scale,
            source=config.download_source,
            hf_repo=config.hf_repo,
        )

    return hr_dir, lr_dir


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


@dataclass
class Div2KConfig:
    """Configuration for preparing DIV2K dataloaders.

    Args:
        root: Base directory where DIV2K will be stored/unpacked.
        scale: Super-resolution scale factor (2, 3, or 4 in the official set).
        subset: "bicubic" (default) or "mild"/"unknown" to pick alternative LR subsets.
        download: Whether to attempt downloading the dataset if missing.
        download_source: "official" to fetch from the ETHZ server or "huggingface" to
            rely on a Hugging Face dataset mirror.
        hf_repo: Hugging Face repo id to pull from when ``download_source`` is
            ``"huggingface"``.
    """

    root: Path
    scale: int = 4
    subset: str = "bicubic"
    download: bool = False
    download_source: str = "official"
    hf_repo: str = "eugenesiow/Div2k"


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


def build_div2k_dataloaders(
    config: Div2KConfig,
    batch_size: int = 4,
    num_workers: int = 0,
    patch_size: Optional[int] = 128,
    augment: bool = True,
) -> Dict[str, DataLoader]:
    """Create train/val dataloaders for DIV2K with optional auto-download.

    This helper wraps :func:`prepare_div2k_folders` and feeds the resolved
    HR/LR directories to :func:`build_dataloaders`, so callers can switch to
    real DIV2K data by toggling a single flag.
    """

    train_hr, train_lr = prepare_div2k_folders(config, split="train")
    val_hr, val_lr = prepare_div2k_folders(config, split="valid")

    train_cfg = DataloaderConfig(
        hr_dir=train_hr,
        lr_dir=train_lr,
        batch_size=batch_size,
        num_workers=num_workers,
        scale=config.scale,
        patch_size=patch_size,
        augment=augment,
    )
    val_cfg = DataloaderConfig(
        hr_dir=val_hr,
        lr_dir=val_lr,
        batch_size=max(1, batch_size // 2),
        num_workers=num_workers,
        scale=config.scale,
        patch_size=None,
        augment=False,
        shuffle=False,
    )

    return build_dataloaders(train_cfg, val_cfg)


def build_loss(config: Optional[LossConfig] = None) -> CompositeLoss:
    return CompositeLoss(config or LossConfig())


@dataclass
class TrainingConfig:
    epochs: int = 1
    lr: float = 1e-4
    device: Optional[str] = None
    log_interval: int = 10
    max_steps_per_epoch: Optional[int] = None
    grad_clip: Optional[float] = None


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: CompositeLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    log_interval: int = 10,
    max_steps: Optional[int] = None,
) -> float:
    model.train()
    running_loss = 0.0
    for step, (lr, hr) in enumerate(dataloader):
        if max_steps is not None and step >= max_steps:
            break
        lr = lr.to(device)
        hr = hr.to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(lr)
        loss, _ = criterion(pred, hr)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (step + 1) % log_interval == 0:
            avg_loss = running_loss / log_interval
            print(f"[train] step {step+1}: loss={avg_loss:.4f}")
            running_loss = 0.0

    return running_loss


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: CompositeLoss,
    device: torch.device,
    max_steps: Optional[int] = None,
) -> float:
    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for step, (lr, hr) in enumerate(dataloader):
            if max_steps is not None and step >= max_steps:
                break
            lr = lr.to(device)
            hr = hr.to(device)
            pred = model(lr)
            loss, _ = criterion(pred, hr)
            losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0


def benchmark_throughput_and_memory(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    warmup: int = 2,
    steps: int = 10,
) -> Dict[str, float]:
    """Return simple throughput/latency/memory measurements."""

    timings: List[float] = []
    memory: float = 0.0
    iterator = iter(dataloader)
    for idx in range(warmup + steps):
        try:
            lr, _ = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            lr, _ = next(iterator)

        lr = lr.to(device)
        torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(lr)
        torch.cuda.synchronize(device) if device.type == "cuda" else None
        duration = time.perf_counter() - start

        if idx >= warmup:
            timings.append(duration)
            if device.type == "cuda":
                memory = max(memory, torch.cuda.max_memory_allocated(device) / (1024**2))
    avg_ms = np.mean(timings) * 1000.0 if timings else 0.0
    throughput = (len(dataloader.dataset) / avg_ms * 1000.0) if avg_ms > 0 else 0.0
    return {
        "latency_ms": avg_ms,
        "throughput_items_per_s": throughput,
        "max_mem_mb": memory,
    }


def run_training(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    config: TrainingConfig,
    loss_config: Optional[LossConfig] = None,
) -> Dict[str, float]:
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    criterion = build_loss(loss_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    history: Dict[str, float] = {}
    for epoch in range(config.epochs):
        print(f"Epoch {epoch + 1}/{config.epochs}")
        train_one_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
            log_interval=config.log_interval,
            max_steps=config.max_steps_per_epoch,
        )
        if "val" in loaders:
            val_loss = validate(
                model,
                loaders["val"],
                criterion,
                device,
                max_steps=config.max_steps_per_epoch,
            )
            history[f"val_loss_epoch_{epoch+1}"] = val_loss
            print(f"[val] loss={val_loss:.4f}")

    history.update(
        benchmark_throughput_and_memory(
            model,
            loaders["train"],
            device,
        )
    )
    return history
