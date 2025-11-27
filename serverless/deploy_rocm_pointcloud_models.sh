#!/bin/bash
# Deploy ROCm-friendly point cloud functions for 3D SIT datasets.
# This script targets cuboid detectors that consume LiDAR (.pcd/.bin) frames
# and return auto-annotations in CVAT's 3D workspace.
#
# Usage:
#   ./deploy_rocm_pointcloud_models.sh [toolbox_name]
#
# Environment overrides:
#   ROCM_USE_TOOLBOX   - Set to "1" to deploy via toolbox enter (default: 0)
#   ROCM_TOOLBOX_NAME  - Name of the toolbox container (default: strix-halo-llm-finetuning)
#   NUCLIO_PROJECT     - Nuclio project (default: cvat)
#   NUCLIO_NETWORK     - Docker network (default: cvat_cvat)

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DEFAULT_TOOLBOX_NAME="strix-halo-llm-finetuning"
TOOLBOX_NAME="${1:-${ROCM_TOOLBOX_NAME:-$DEFAULT_TOOLBOX_NAME}}"

DEFAULT_DEPLOYER="$SCRIPT_DIR/deploy_rocm_host.sh"
if [ "${ROCM_USE_TOOLBOX:-0}" = "1" ]; then
    DEFAULT_DEPLOYER="$SCRIPT_DIR/deploy_rocm_toolbox.sh"
fi
DEPLOYER="$DEFAULT_DEPLOYER"

if [ ! -x "$DEPLOYER" ]; then
    echo "Expected executable helper at $DEPLOYER" >&2
    exit 1
fi

deploy_pointcloud() {
    local label="$1"
    local path="$2"

    if [ ! -d "$path" ]; then
        echo "Skipping $label (directory $path not found)"
        return 0
    fi

    if find "$path" -name "function-rocm.yaml" -print -quit >/dev/null; then
        echo "Deploying $label via $DEPLOYER"
        if [ "${ROCM_USE_TOOLBOX:-0}" = "1" ]; then
            "$DEPLOYER" "$path" "$TOOLBOX_NAME"
        else
            "$DEPLOYER" "$path"
        fi
    else
        echo "Skipping $label (no function-rocm.yaml found under $path)"
    fi
}

deploy_pointcloud "SIT PointCloud Cuboids (ROCm)" \
    "$SCRIPT_DIR/pytorch/open3d/sit_pointcloud"

cat <<'EOF'
Deployed 3D point-cloud cuboid detector(s). Use the CVAT Models tab to confirm
that "SIT PointCloud Cuboids (ROCm)" is available, then open a 3D job and run
Automatic Annotation. This model exposes "car", "truck", and "pedestrian"
labels; map them to your SIT task labels inside the annotation dialog.
EOF

