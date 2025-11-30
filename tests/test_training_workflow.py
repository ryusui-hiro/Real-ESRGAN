import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from efficient_sr import EfficientSRNet
from training_utils import (
    Div2KConfig,
    TrainingConfig,
    build_div2k_dataloaders,
    run_training,
)


def _create_dummy_div2k(root: Path, scale: int = 4):
    train_hr = root / "DIV2K_train_HR"
    train_lr = root / f"DIV2K_train_LR_bicubic_X{scale}"
    val_hr = root / "DIV2K_valid_HR"
    val_lr = root / f"DIV2K_valid_LR_bicubic_X{scale}"
    for folder in (train_hr, train_lr, val_hr, val_lr):
        folder.mkdir(parents=True, exist_ok=True)
    for idx in range(2):
        hr_img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
        Image.fromarray(hr_img).save(train_hr / f"{idx:04d}.png")
        Image.fromarray(hr_img[::scale, ::scale]).save(train_lr / f"{idx:04d}.png")
        Image.fromarray(hr_img).save(val_hr / f"val{idx:04d}.png")
        Image.fromarray(hr_img[::scale, ::scale]).save(val_lr / f"val{idx:04d}.png")


def test_build_div2k_dataloaders(tmp_path: Path):
    _create_dummy_div2k(tmp_path)
    cfg = Div2KConfig(root=tmp_path, download=False)
    loaders = build_div2k_dataloaders(cfg, batch_size=1, patch_size=32, augment=False)
    lr, hr = next(iter(loaders["train"]))
    assert lr.shape[-1] == hr.shape[-1] // cfg.scale
    assert lr.shape[-2] == hr.shape[-2] // cfg.scale


def test_run_training_with_dummy_data(tmp_path: Path):
    _create_dummy_div2k(tmp_path)
    cfg = Div2KConfig(root=tmp_path, download=False)
    loaders = build_div2k_dataloaders(cfg, batch_size=1, patch_size=32, augment=False)

    model = EfficientSRNet(upscale=cfg.scale)
    history = run_training(
        model,
        loaders,
        TrainingConfig(epochs=1, max_steps_per_epoch=1, log_interval=1, device="cpu"),
        loss_config=None,
    )
    assert "latency_ms" in history
    assert "throughput_items_per_s" in history
