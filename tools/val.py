#!/usr/bin/env python3
"""Validate an LCD-YOLO checkpoint with the recorded checkpoint protocol."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Checkpoint path")
    parser.add_argument("--data", required=True, help="Dataset YAML path")
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default="0", help="CUDA device index or 'cpu'")
    parser.add_argument("--project", default=str(ROOT / "runs/val"))
    parser.add_argument("--name", default="lcd-yolo")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=0.001,
        iou=0.5,
        max_det=300,
        rect=True,
        workers=0,
        device=args.device,
        project=args.project,
        name=args.name,
    )
    box = metrics.box
    print(
        {
            "precision": float(box.mp),
            "recall": float(box.mr),
            "map50": float(box.map50),
            "map50_95": float(box.map),
        }
    )


if __name__ == "__main__":
    main()
