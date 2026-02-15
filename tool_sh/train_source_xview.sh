#!/bin/bash

set -euo pipefail

METHOD="smallbackbone6"
CONFIG_FILE="Omni_Tuner_configs/swin-l_xview3c/xview3c_retinanet_swin_large_1x_${METHOD}.py"
PRETRAINED_PATH="model/swin_large_patch4_window7_224_22k.pth"

CUDA_VISIBLE_DEVICES=3 python train_source.py "$CONFIG_FILE" --cfg-options model.pretrained="$PRETRAINED_PATH"
