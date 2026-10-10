"""Smoke tests for the public LCD-YOLO configuration."""

from pathlib import Path

import pytest
import torch

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "ultralytics/cfg/models/11/lcd-yolo.yaml"
ABLATIONS = {
    "lcd-yolo-lcat-to-c3k2.yaml": 2_900_489,
    "lcd-yolo-msblock-to-c3k2.yaml": 3_294_521,
    "lcd-yolo-no-cmca.yaml": 3_602_223,
}


def test_lcd_yolo_build_and_forward() -> None:
    model = YOLO(str(CONFIG)).model.eval()
    layer_names = [layer.__class__.__name__ for layer in model.model]

    assert layer_names.count("SwinTransformerBlock_v2a") == 2
    assert layer_names.count("SM_block_v2a") == 2
    assert layer_names.count("MSBlock") == 2
    assert layer_names.count("Detect") == 1
    assert sum(parameter.numel() for parameter in model.parameters()) == 3_744_777

    with torch.no_grad():
        output = model(torch.zeros(1, 3, 160, 160))
    assert output is not None


@pytest.mark.parametrize(("filename", "expected_params"), ABLATIONS.items())
def test_paper_ablation_build_and_forward(filename: str, expected_params: int) -> None:
    config = ROOT / "ultralytics/cfg/models/11/ablations" / filename
    model = YOLO(str(config)).model.eval()

    assert sum(parameter.numel() for parameter in model.parameters()) == expected_params
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 160, 160))
    assert output is not None
