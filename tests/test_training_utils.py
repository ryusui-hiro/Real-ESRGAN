import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from training_utils import (
    CompositeLoss,
    DataloaderConfig,
    LossConfig,
    SuperResolutionDataset,
    build_dataloaders,
    build_loss,
)


def _write_image(path: Path, size: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    array = (rng.random((size, size, 3)) * 255).astype(np.uint8)
    Image.fromarray(array).save(path)


def test_dataset_generates_lr_when_missing(tmp_path: Path) -> None:
    hr_dir = tmp_path / "hr"
    hr_dir.mkdir()
    _write_image(hr_dir / "sample.png", size=32, seed=0)

    random.seed(42)
    torch.manual_seed(0)
    dataset = SuperResolutionDataset(hr_dir=hr_dir, lr_dir=None, scale=2, patch_size=16, augment=False)

    lr, hr = dataset[0]
    assert hr.shape == (3, 16, 16)
    assert lr.shape == (3, 8, 8)
    assert torch.isfinite(lr).all()


def test_dataset_reads_paired_lr(tmp_path: Path) -> None:
    hr_dir = tmp_path / "hr"
    lr_dir = tmp_path / "lr"
    hr_dir.mkdir()
    lr_dir.mkdir()
    _write_image(hr_dir / "paired.png", size=40, seed=1)
    _write_image(lr_dir / "paired.png", size=20, seed=2)

    dataset = SuperResolutionDataset(hr_dir=hr_dir, lr_dir=lr_dir, scale=2, patch_size=None, augment=False)
    lr, hr = dataset[0]
    assert hr.shape == (3, 40, 40)
    assert lr.shape == (3, 20, 20)



def test_composite_loss_weights_components() -> None:
    loss = build_loss(LossConfig(pixel_weight=1.0, charbonnier_weight=0.5, tv_weight=0.25, charbonnier_epsilon=1e-6))
    pred = torch.ones(1, 3, 4, 4)
    target = torch.zeros_like(pred)

    total, components = loss(pred, target)

    assert set(components.keys()) == {"l1", "charbonnier", "tv"}
    expected = components["l1"] * 1.0 + components["charbonnier"] * 0.5 + components["tv"] * 0.25
    assert torch.allclose(total, expected)


def test_dataloader_builder_returns_loaders(tmp_path: Path) -> None:
    hr_dir = tmp_path / "train_hr"
    hr_dir.mkdir()
    _write_image(hr_dir / "sample.png", size=16, seed=3)

    train_config = DataloaderConfig(hr_dir=hr_dir, batch_size=1, patch_size=None, shuffle=False, num_workers=0)
    loaders = build_dataloaders(train_config)
    assert "train" in loaders
    batch = next(iter(loaders["train"]))
    lr, hr = batch
    assert lr.shape == (1, 3, 4, 4)
    assert hr.shape == (1, 3, 16, 16)

    with torch.no_grad():
        loss_module = CompositeLoss(LossConfig())
        total, comps = loss_module(hr, hr)
        assert torch.isclose(total, torch.tensor(0.0)).all()
        assert comps["l1"] == 0
