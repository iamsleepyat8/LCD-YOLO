#!/usr/bin/env python3
"""Train LCD-YOLO with the recorded scratch-training defaults."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "ultralytics/cfg/models/11/lcd-yolo.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Model YAML path")
    parser.add_argument("--data", required=True, help="Dataset YAML path")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0", help="CUDA device index or 'cpu'")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default=str(ROOT / "runs/detect"))
    parser.add_argument("--name", default="lcd-yolo")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        patience=0,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        pretrained=False,
        optimizer="auto",
        seed=0,
        deterministic=True,
        cos_lr=True,
        close_mosaic=10,
        amp=True,
        lr0=0.01,
        lrf=0.01,
        weight_decay=0.0005,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.3,
    )


if __name__ == "__main__":
    main()
