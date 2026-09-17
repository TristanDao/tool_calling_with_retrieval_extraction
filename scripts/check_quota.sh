#!/bin/bash
CONDA_BIN="/home/thinh/.conda/envs/ai/bin"
echo "=== KAGGLE ACCELERATOR QUOTA (GPU / TPU) ==="
$CONDA_BIN/kaggle quota
$CONDA_BIN/python scripts/check_quota.py
