#!/usr/bin/env python3
"""Build LCD-YOLO and print its model summary."""

from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "ultralytics/cfg/models/11/lcd-yolo.yaml"


if __name__ == "__main__":
    YOLO(str(MODEL)).info(detailed=False, verbose=True, imgsz=640)
