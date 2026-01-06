#!/bin/bash
# Deploy/Stop ROCm-friendly point cloud functions for 3D SIT datasets.
# This script targets cuboid detectors that consume LiDAR (.pcd/.bin) frames
# and return auto-annotations in CVAT's 3D workspace.
#
# Usage:
#   ./deploy_rocm_pointcloud_models.sh <model_name> [action]
#
# Models:
#   fcaf3d     - FCAF3D 3D Cuboids (state-of-the-art indoor detection)
#   sit        - SIT PointCloud Cuboids (heuristic-based detector)
#
# Actions:
#   deploy     - Deploy model(s) (default)
#   stop       - Stop specific model
#   stop-all   - Stop all point cloud models
#
# Environment overrides:
#   NUCLIO_PROJECT     - Nuclio project (default: cvat)
#   NUCLIO_NETWORK     - Docker network (default: cvat_cvat)
#   FCAF3D_USE_ROCM    - Set to "1" to deploy FCAF3D with ROCm (default: 0)

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
MODEL_NAME="${1:-fcaf3d}"
ACTION="${2:-deploy}"

DEPLOYER="$SCRIPT_DIR/deploy_rocm_host.sh"

if [ ! -x "$DEPLOYER" ]; then
    echo "Expected executable helper at $DEPLOYER" >&2
    exit 1
fi

# Function to get function name from config
get_function_name() {
    local config_file="$1"
    local func_name=""

    # Extract function name from metadata.name in YAML
    if [ -f "$config_file" ]; then
        func_name=$(grep -A 10 "^metadata:" "$config_file" | grep "name:" | head -1 | sed 's/.*name:\s*//' | tr -d ' ')
    fi

    echo "$func_name"
}

# Stop a specific function
stop_function() {
    local func_name="$1"

    if [ -z "$func_name" ]; then
        echo "Error: Function name not provided"
        return 1
    fi

    echo "Stopping function: $func_name"
    if nuctl get function "$func_name" >/dev/null 2>&1; then
        nuctl delete function "$func_name"
        echo "✓ Successfully stopped $func_name"
    else
        echo "⚠ Function $func_name not found or already stopped"
    fi
}

# Stop all point cloud functions
stop_all_functions() {
    echo "Stopping all point cloud functions..."

    # Common function names for point cloud models
    local functions=(
        "pth-mmdet3d-fcaf3d-rocm"
        "pth-mmdet3d-fcaf3d-cpu"
        "pth-open3d-sit-pointcloud-rocm"
        "pth-open3d-sit-pointcloud-cpu"
    )

    for func in "${functions[@]}"; do
        if nuctl get function "$func" >/dev/null 2>&1; then
            echo "Stopping $func..."
            nuctl delete function "$func"
            echo "✓ Stopped $func"
        fi
    done

    echo "✓ All point cloud functions stopped"
}

deploy_pointcloud() {
    local label="$1"
    local path="$2"

    if [ ! -d "$path" ]; then
        echo "Error: Directory $path not found for $label"
        return 1
    fi

    # Check for function config files
    local config_found=false
    local config_file=""

    # Priority: ROCm -> CPU -> generic
    if find "$path" -name "function-rocm.yaml" -print -quit >/dev/null 2>/dev/null; then
        config_found=true
        config_file="$path/function-rocm.yaml"
        echo "Found ROCm configuration for $label"
    elif find "$path" -name "function-cpu.yaml" -print -quit >/dev/null 2>/dev/null; then
        config_found=true
        config_file="$path/function-cpu.yaml"
        echo "Found CPU configuration for $label"
    elif find "$path" -name "function*.yaml" -print -quit >/dev/null 2>/dev/null; then
        config_found=true
        config_file=$(find "$path" -name "function*.yaml" | head -1)
        echo "Found generic configuration for $label"
    fi

    if [ "$config_found" = true ]; then
        echo "Deploying $label via $DEPLOYER (config: $config_file)"
        "$DEPLOYER" "$path" "$(basename "$config_file")"
        echo "✓ Successfully deployed $label"
    else
        echo "Error: No function config found under $path"
        return 1
    fi
}

# Main logic
case "$ACTION" in
    "deploy")
        case "$MODEL_NAME" in
            "fcaf3d")
                if [ "${FCAF3D_USE_ROCM:-0}" = "1" ]; then
                    deploy_pointcloud "FCAF3D 3D Cuboids (ROCm)" \
                        "$SCRIPT_DIR/pytorch/mmdetection3d/fcaf3d"
                else
                    echo "Error: FCAF3D only supports ROCm deployment now. Use: FCAF3D_USE_ROCM=1 $0 fcaf3d deploy"
                    exit 1
                fi
                ;;
            "sit")
                deploy_pointcloud "SIT PointCloud Cuboids (ROCm)" \
                    "$SCRIPT_DIR/pytorch/open3d/sit_pointcloud"
                ;;
            "all")
                echo "Error: 'all' deployment is not supported. Deploy models individually:"
                echo "  FCAF3D_USE_ROCM=1 $0 fcaf3d deploy  # Deploy FCAF3D"
                echo "  $0 sit deploy                        # Deploy SIT"
                exit 1
                ;;
            *)
                echo "Error: Unknown model '$MODEL_NAME'"
                echo "Available models: fcaf3d, sit"
                echo "Usage: $0 <model_name> [deploy|stop]  # or: $0 <any_name> stop-all"
                exit 1
                ;;
        esac
        ;;

    "stop")
        case "$MODEL_NAME" in
            "fcaf3d")
                if [ "${FCAF3D_USE_ROCM:-0}" = "1" ]; then
                    stop_function "pth-mmdet3d-fcaf3d-rocm"
                else
                    stop_function "pth-mmdet3d-fcaf3d-cpu"
                fi
                ;;
            "sit")
                stop_function "pth-open3d-sit-pointcloud-rocm"
                ;;
            *)
                echo "Error: Unknown model '$MODEL_NAME'"
                echo "Available models: fcaf3d, sit"
                exit 1
                ;;
        esac
        ;;

    "stop-all")
        stop_all_functions
        ;;

    *)
        echo "Error: Unknown action '$ACTION'"
        echo "Available actions: deploy, stop, stop-all"
        echo "Usage: $0 <model_name> [deploy|stop]  # or: $0 <any_name> stop-all"
        exit 1
        ;;
esac

# Success message for deploy action
if [ "$ACTION" = "deploy" ]; then
    cat <<EOF

✓ Successfully deployed $MODEL_NAME model. Use the CVAT Models tab to confirm
that the model is available, then open a 3D job and run Automatic Annotation.

Model Details:
EOF

    case "$MODEL_NAME" in
        "fcaf3d")
            echo "- FCAF3D 3D Cuboids: State-of-the-art indoor detection (64.3% mAP)"
            ;;
        "sit")
            echo "- SIT PointCloud Cuboids: Heuristic-based detector for social navigation"
            ;;
    esac

    cat <<EOF
Both models expose pedestrian detection labels; map them to your task labels
inside the annotation dialog.
EOF
fi

