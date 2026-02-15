#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${REPO_ROOT}}"
METHOD="smallbackbone6"

CONFIG="${PROJECT_ROOT}/Omni_Tuner_configs/swin-l_gtav10k1c/gtav10k1c_retinanet_swin_large_1x_${METHOD}.py"
WORK_DIR="${PROJECT_ROOT}/work_dirs/gtav10k1c_retinanet_swin_large_1x_${METHOD}"
CHECKPOINT="${WORK_DIR}/best.pth"

DEVICE="${DEVICE:-cuda:2}"
BATCH_SIZE="${BATCH_SIZE:-16}"
WORKERS="${WORKERS:-8}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

STYLE_OUT="${WORK_DIR}/style_stats_patch_stage0123_prepost.pth"
SCREEN_OUT="${WORK_DIR}/gtav10k1c_prototypes_screen.pth"
SCREEN_TOKENS="${WORK_DIR}/gtav10k1c_tokens_screen.pth"
FREQ_OUT="${WORK_DIR}/gtav10k1c_prototypes_backbone_frequency.pth"
FREQ_TOKENS="${WORK_DIR}/gtav10k1c_tokens_backbone_frequency.pth"
GLOBAL_OUT="${WORK_DIR}/gtav10k1c_prototypes_global.pth"
GLOBAL_TOKENS="${WORK_DIR}/gtav10k1c_tokens_global.pth"

CMD=(
    python "${PROJECT_ROOT}/prototype_tools/extract_source_all_in_one.py"
    --config "${CONFIG}"
    --checkpoint "${CHECKPOINT}"
    --device "${DEVICE}"
    --batch-size "${BATCH_SIZE}"
    --workers "${WORKERS}"
    --style-out "${STYLE_OUT}"
    --screen-out "${SCREEN_OUT}"
    --screen-save-tokens "${SCREEN_TOKENS}"
    --freq-out "${FREQ_OUT}"
    --freq-save-tokens "${FREQ_TOKENS}"
    --global-out "${GLOBAL_OUT}"
    --global-save-tokens "${GLOBAL_TOKENS}"
    --global-feature-source c4
    --global-pool-type avg
    --freq-feature-stage c5
    --freq-pool-size 24
    --freq-channel-aggregation mean
    --freq-num-prototypes 16
    --screen-num-prototypes 3
    --screen-sinkhorn-epochs 100
    --freq-sinkhorn-epochs 120
    --global-sinkhorn-epochs 100
    --screen-sinkhorn-batch-size 512
    --freq-sinkhorn-batch-size 512
    --global-sinkhorn-batch-size 512
    --screen-sinkhorn-queue-size 8192
    --freq-sinkhorn-queue-size 8192
    --global-sinkhorn-queue-size 8192
    --screen-sinkhorn-momentum 0.02
    --freq-sinkhorn-momentum 0.02
    --global-sinkhorn-momentum 0.02
    --screen-sinkhorn-iterations 5
    --freq-sinkhorn-iterations 5
    --global-sinkhorn-iterations 5
    --screen-sinkhorn-epsilon 1e-2
    --freq-sinkhorn-epsilon 1e-2
    --global-sinkhorn-epsilon 1e-2
    --no-visualize
)

if [[ -n "${MAX_SAMPLES}" ]]; then
    CMD+=(--max-samples "${MAX_SAMPLES}")
fi

"${CMD[@]}"
