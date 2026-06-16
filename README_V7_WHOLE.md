# Mahjong V7 Whole Code Package

This package contains V7-related code and small metadata only. It intentionally excludes model weights (`*.pkl`).

## Main V7 model

The V7 calibrated model uses `model_v6.CNNModel` architecture and expects the weight file separately at:

`data/mahjong_v7_calibrated_best.pkl`

Remote original weight path, not included here:

`/root/majong/SL/model/checkpoint/v7_calibrated_095/mahjong_v7_calibrated_best.pkl`

## Directories

- `source/`: training, evaluation, model, feature, and arena source files.
- `botzone_storage/`: Botzone-ready Python files without model weights.
- `metadata/`: run configs, metrics, diagnostics, and self-play summaries.
