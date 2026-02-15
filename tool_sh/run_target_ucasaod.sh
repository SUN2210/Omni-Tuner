#!/bin/bash

set -euo pipefail

METHOD="smallbackbone6"
STYLE_MOMENTUM="1"
STYLE_APPLY_TO="patch_embed"
STYLE_MIX_ALPHA="x"
STYLE_MIX_ALPHA_INIT="0.2"
STYLE_MIX_ALPHA_SCOPE="shared"
MIX_ALPHA_LR_MULT="200"
STYLE_PROPAGATION="full"

CONFIG_FILE="Omni_Tuner_configs/swin-l_ucasaod1c_fs/ucasaod1c_fs_retinanet_swin_large_5x_${METHOD}.py"
PRETRAINED_PATH="work_dirs/gtav10k1c_retinanet_swin_large_1x_${METHOD}/best.pth"
STYLE_STATS_PATH="work_dirs/gtav10k1c_retinanet_swin_large_1x_${METHOD}/style_stats_patch_stage0123_prepost.pth"

if [ ! -f "$PRETRAINED_PATH" ]; then
    echo "Missing pretrained checkpoint: $PRETRAINED_PATH"
    exit 1
fi

CFG_OPTIONS=(load_from="$PRETRAINED_PATH" model.backbone.style_adapter.mix_alpha_lr_mult="$MIX_ALPHA_LR_MULT")

if [ -f "$STYLE_STATS_PATH" ]; then
    CUDA_VISIBLE_DEVICES=2 python train.py "$CONFIG_FILE" \
        --style-stats "$STYLE_STATS_PATH" \
        --style-momentum "$STYLE_MOMENTUM" \
        --style-apply-to "$STYLE_APPLY_TO" \
        --style-mix-alpha "$STYLE_MIX_ALPHA" \
        --style-mix-alpha-init "$STYLE_MIX_ALPHA_INIT" \
        --style-mix-alpha-scope "$STYLE_MIX_ALPHA_SCOPE" \
        --style-propagation "$STYLE_PROPAGATION" \
        --cfg-options "${CFG_OPTIONS[@]}"
else
    CUDA_VISIBLE_DEVICES=2 python train.py "$CONFIG_FILE" --cfg-options "${CFG_OPTIONS[@]}"
fi
