# LCD-YOLO

[English](README.md) | 简体中文

LCD-YOLO 是一个面向施工安全场景、以 YOLO11n 为基础构建的目标检测模型。

论文名称与代码中的历史类名对应如下：

- CMCA：`SM_block_v2a`
- LCAT：`SwinTransformerBlock_v2a`
- DS-MSBlock：`MSBlock`

为兼容已有 checkpoint，代码保留了这些历史类名。

## 安装

```bash
conda env create -f environment.yml
conda activate lcd-yolo
python -m pip install -e .
```


## 训练

```bash
python tools/train.py \
  --data /path/to/construction-safety.yaml \
  --device 0
```

默认参数与最终 LCD-YOLO 实验记录一致：从头训练 300 epochs、图像尺寸 640、
batch size 64、seed 0、确定性训练、余弦学习率调度，并使用 Ultralytics 的
自动优化器选择。

如需运行论文中的消融实验，可通过 `--model` 指定：

```text
ultralytics/cfg/models/11/ablations/lcd-yolo-lcat-to-c3k2.yaml
ultralytics/cfg/models/11/ablations/lcd-yolo-msblock-to-c3k2.yaml
ultralytics/cfg/models/11/ablations/lcd-yolo-no-cmca.yaml
```

## 验证 checkpoint

```bash
python tools/val.py \
  --weights /path/to/best.pt \
  --data /path/to/construction-safety.yaml \
  --device 0
```

## 查看模型与运行测试

```bash
python tools/model_info.py
pytest -q tests/test_lcd_yolo.py
```

## 许可证与上游项目

本项目派生自 Ultralytics，并依据 AGPL-3.0 许可证发布。详情请参阅
`LICENSE` 和 `UPSTREAM.md`。
