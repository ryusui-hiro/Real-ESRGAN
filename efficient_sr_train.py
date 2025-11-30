"""CLI for EfficientSR training with real DIV2K data support."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch

from efficient_sr import EfficientSRNet
from training_utils import (
    DataloaderConfig,
    Div2KConfig,
    LossConfig,
    TrainingConfig,
    build_dataloaders,
    build_div2k_dataloaders,
    run_training,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EfficientSR")
    parser.add_argument("--dataset", choices=["custom", "div2k"], default="custom")
    parser.add_argument("--hr-dir", type=Path, help="HR image directory (custom dataset)")
    parser.add_argument("--lr-dir", type=Path, default=None, help="LR image directory (custom dataset)")
    parser.add_argument("--div2k-root", type=Path, default=Path("data/DIV2K"), help="Base folder for DIV2K")
    parser.add_argument("--div2k-download", action="store_true", help="Download DIV2K if missing")
    parser.add_argument("--div2k-source", choices=["official", "huggingface"], default="official")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4, dest="learning_rate")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=None, help="Max steps per epoch (for quick smoke tests)")
    parser.add_argument("--device", type=str, default=None, help="Device string (cuda or cpu)")
    parser.add_argument("--no-augment", action="store_true", help="Disable train-time flips")
    return parser.parse_args()


def _build_loaders(args: argparse.Namespace):
    if args.dataset == "div2k":
        div2k_cfg = Div2KConfig(
            root=args.div2k_root,
            scale=args.scale,
            download=args.div2k_download,
            download_source=args.div2k_source,
        )
        return build_div2k_dataloaders(
            div2k_cfg,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            augment=not args.no_augment,
        )

    if args.hr_dir is None:
        raise ValueError("--hr-dir is required for custom datasets")
    train_conf = DataloaderConfig(
        hr_dir=args.hr_dir,
        lr_dir=args.lr_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        scale=args.scale,
        augment=not args.no_augment,
    )
    return build_dataloaders(train_conf)


def main(args: Optional[argparse.Namespace] = None) -> None:
    args = args or parse_args()
    loaders = _build_loaders(args)

    model = EfficientSRNet(upscale=args.scale)
    tconfig = TrainingConfig(
        epochs=args.epochs,
        lr=args.learning_rate,
        device=args.device,
        log_interval=args.log_interval,
        max_steps_per_epoch=args.max_steps,
    )
    history = run_training(model, loaders, tconfig, LossConfig())
    print("Benchmark summary:", history)


if __name__ == "__main__":
    main()
