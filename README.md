# LCD-YOLO

[English](README.md) | [简体中文](README.zh-CN.md)

LCD-YOLO is a YOLO11n-derived detector for construction-safety scenes.

The public paper names map to the implementation as follows:

- CMCA: `SM_block_v2a`
- LCAT: `SwinTransformerBlock_v2a`
- DS-MSBlock: `MSBlock`

The legacy Python class names are retained so that existing checkpoints remain
loadable.

## Installation

```bash
conda env create -f environment.yml
conda activate lcd-yolo
python -m pip install -e .
```

## Train

```bash
python tools/train.py \
  --data /path/to/construction-safety.yaml \
  --device 0
```

The defaults reproduce the recorded LCD-YOLO training configuration: training
from scratch for 300 epochs, image size 640, batch size 64, seed 0, deterministic
mode, cosine learning-rate scheduling, and Ultralytics automatic optimizer
selection.

## Validate a checkpoint

```bash
python tools/val.py \
  --weights /path/to/best.pt \
  --data /path/to/construction-safety.yaml \
  --device 0
```

## Inspect and test the model

```bash
python tools/model_info.py
pytest -q tests/test_lcd_yolo.py
```

## License and upstream

This code is derived from Ultralytics and distributed under AGPL-3.0. See
`LICENSE` and `UPSTREAM.md`.
