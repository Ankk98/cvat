#!/bin/bash
# Curated ROCm deployments for robotics/VLA annotation workloads.
# This script deploys a balanced set of functions that cover the common
# auto-annotation use cases in CVAT: segmentation, detection, and tracking.
# By default it uses the host-based deployer, but you can switch back to
# the toolbox helper by setting ROCM_USE_TOOLBOX=1 or overriding ROCM_DEPLOYER.
#
# Usage:
#   ./deploy_rocm_robotics_models.sh [toolbox_name_if_needed]
# Environment overrides:
#   ROCM_DEPLOYER     - Path to a custom deploy script (default: deploy_rocm_host.sh)
#   ROCM_USE_TOOLBOX  - Set to 1 to force deploy_rocm_toolbox.sh
#   ROCM_TOOLBOX_NAME - Toolbox container name (default: strix-halo-llm-finetuning)

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DEFAULT_DEPLOYER="$SCRIPT_DIR/deploy_rocm_host.sh"
if [ "${ROCM_USE_TOOLBOX:-0}" = "1" ]; then
    DEFAULT_DEPLOYER="$SCRIPT_DIR/deploy_rocm_toolbox.sh"
fi
DEPLOYER="${ROCM_DEPLOYER:-$DEFAULT_DEPLOYER}"
EXTRA_ARGS=()
if [ "$DEPLOYER" = "$SCRIPT_DIR/deploy_rocm_toolbox.sh" ]; then
    DEFAULT_TOOLBOX_NAME="strix-halo-llm-finetuning"
    TOOLBOX_NAME="${1:-${ROCM_TOOLBOX_NAME:-$DEFAULT_TOOLBOX_NAME}}"
    EXTRA_ARGS=("$TOOLBOX_NAME")
fi
shopt -s globstar nullglob

if [ ! -x "$DEPLOYER" ]; then
    echo "Expected helper script at $DEPLOYER. Make sure it is executable." >&2
    exit 1
fi

deploy_model() {
    local label="$1"
    local path="$2"
    local has_config=0

    if [ ! -d "$path" ]; then
        echo "Skipping $label (path \"$path\" not found)"
        return 0
    fi

    for config in "$path"/**/function-rocm.yaml "$path"/**/function-gpu.yaml; do
        if [ -f "$config" ]; then
            has_config=1
            break
        fi
    done

    if [ "$has_config" -eq 0 ]; then
        echo "Skipping $label (no function-rocm.yaml or function-gpu.yaml under $path)"
        return 0
    fi

    echo "Deploying $label via $DEPLOYER..."
    "$DEPLOYER" "$path" "${EXTRA_ARGS[@]}"
}

# Segmentation (interactive masks)
segmentation_primary="$SCRIPT_DIR/pytorch/facebookresearch/sam"

# Detection (multi-class bounding boxes)
detection_primary="$SCRIPT_DIR/pytorch/facebookresearch/detectron2/retinanet_r101"

# Tracking (long-lived IDs for robots/parts)
tracking_primary="$SCRIPT_DIR/pytorch/dschoerk/transt"

deploy_model "Segment Anything (segmentation)" "$segmentation_primary"
deploy_model "Detectron2 RetinaNet R101 (detection)" "$detection_primary"
deploy_model "Transt Transformer Tracker (tracking)" "$tracking_primary"

cat <<'EOF'
# Additional curated options (uncomment to enable):
# segmentation_alt="$SCRIPT_DIR/pytorch/facebookresearch/sam"            # e.g., swap with SAM2 once added
# detection_alt="$SCRIPT_DIR/openvino/omz/public/yolo-v3-tf"             # fallback detector (CPU/OpenVINO)
# tracking_alt="$SCRIPT_DIR/pytorch/foolwood/siammask"                   # SiamMask tracker
# vision_language_primary="/path/to/custom/vla/nuclio"                   # Add your LLaVA/Qwen-VL Nuclio path here
# "$DEPLOYER" "$segmentation_alt" "${EXTRA_ARGS[@]}"
# "$DEPLOYER" "$detection_alt" "${EXTRA_ARGS[@]}"
# "$DEPLOYER" "$tracking_alt" "${EXTRA_ARGS[@]}"
# "$DEPLOYER" "$vision_language_primary" "${EXTRA_ARGS[@]}"             # Once a VLA function is prepared
EOF

