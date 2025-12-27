#!/bin/bash
# Deploy egocentric vision models for hand-pose skeleton, semantic segmentation,
# and instance segmentation on AMD/ROCm devices.
#
# This script provides a convenient way to deploy models optimized for first-person
# view datasets, focusing on hand tracking, interactive segmentation, and object detection.
#
# Usage:
#   ./deploy_egocentric_models.sh [options]
#
# Options:
#   --all           Deploy all available egocentric models (default)
#   --sam           Deploy SAM for interactive segmentation only
#   --sam-auto      Deploy SAM Auto for automatic segmentation only
#   --detectron2    Deploy Detectron2 RetinaNet for instance segmentation only
#   --mmpose        Deploy MMPose for hand pose estimation only
#   --cpu           Use CPU deployment instead of ROCm (for models without ROCm support)
#   --toolbox       Use toolbox deployment instead of host deployment
#   --toolbox-name NAME  Specify toolbox name (default: strix-halo-llm-finetuning)
#   --help          Show this help message
#
# Environment variables:
#   EGOCENTRIC_TOOLBOX_NAME  - Default toolbox name
#   EGOCENTRIC_USE_TOOLBOX   - Set to 1 to force toolbox deployment
#   EGOCENTRIC_USE_CPU       - Set to 1 to force CPU deployment
#
# Examples:
#   ./deploy_egocentric_models.sh                    # Deploy all models with ROCm
#   ./deploy_egocentric_models.sh --sam --cpu       # Deploy only SAM on CPU
#   ./deploy_egocentric_models.sh --toolbox --toolbox-name my-toolbox

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Default configuration
DEPLOY_ALL=true
DEPLOY_SAM=false
DEPLOY_SAM_AUTO=false
DEPLOY_DETECTRON2=false
DEPLOY_MMPOSE=false
DEPLOY_MEDIAPIPE=false
USE_ROCM=true
USE_TOOLBOX=false
TOOLBOX_NAME="${EGOCENTRIC_TOOLBOX_NAME:-strix-halo-llm-finetuning}"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Show usage information
show_help() {
    cat << EOF
Deploy egocentric vision models for hand-pose skeleton, semantic segmentation,
and instance segmentation on AMD/ROCm devices.

USAGE:
    $0 [OPTIONS]

OPTIONS:
    --all                   Deploy all available egocentric models (default)
    --sam                   Deploy SAM for interactive segmentation only
    --sam-auto              Deploy SAM Auto for automatic segmentation only
    --detectron2            Deploy Detectron2 RetinaNet for instance segmentation only
    --mmpose                Deploy MMPose for hand pose estimation only
    --mediapipe             Deploy MediaPipe for lightweight pose estimation only
    --cpu                   Use CPU deployment instead of ROCm
    --toolbox               Use toolbox deployment instead of host deployment
    --toolbox-name NAME     Specify toolbox name (default: strix-halo-llm-finetuning)
    --help, -h              Show this help message

ENVIRONMENT VARIABLES:
    EGOCENTRIC_TOOLBOX_NAME     Default toolbox name
    EGOCENTRIC_USE_TOOLBOX      Set to 1 to force toolbox deployment
    EGOCENTRIC_USE_CPU          Set to 1 to force CPU deployment

EXAMPLES:
    $0                              # Deploy all models with ROCm
    $0 --sam --cpu                  # Deploy only SAM on CPU
    $0 --toolbox --toolbox-name my-toolbox
    $0 --mmpose --detectron2        # Deploy specific models

MODELS INCLUDED:
    • SAM (Segment Anything) - Interactive segmentation for precise hand/object annotation
    • Detectron2 RetinaNet R101 - Instance segmentation for egocentric scenes
    • MMPose HRNet-W32 - Hand pose estimation for first-person view tracking
    • MediaPipe Pose - Lightweight 33-keypoint pose estimation (fast CPU inference)

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            DEPLOY_ALL=true
            shift
            ;;
        --sam)
            DEPLOY_SAM=true
            DEPLOY_ALL=false
            shift
            ;;
        --sam-auto)
            DEPLOY_SAM_AUTO=true
            DEPLOY_ALL=false
            shift
            ;;
        --detectron2)
            DEPLOY_DETECTRON2=true
            DEPLOY_ALL=false
            shift
            ;;
        --mmpose)
            DEPLOY_MMPOSE=true
            DEPLOY_ALL=false
            shift
            ;;
        --mediapipe)
            DEPLOY_MEDIAPIPE=true
            DEPLOY_ALL=false
            shift
            ;;
        --cpu)
            USE_ROCM=false
            shift
            ;;
        --toolbox)
            USE_TOOLBOX=true
            shift
            ;;
        --toolbox-name)
            TOOLBOX_NAME="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            log_error "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check environment variables
if [[ "${EGOCENTRIC_USE_TOOLBOX:-0}" = "1" ]]; then
    USE_TOOLBOX=true
fi

if [[ "${EGOCENTRIC_USE_CPU:-0}" = "1" ]]; then
    USE_ROCM=false
fi

# Determine deployment method
if [[ "$USE_TOOLBOX" = true ]]; then
    DEPLOYER="$SCRIPT_DIR/deploy_rocm_toolbox.sh"
    EXTRA_ARGS=("$TOOLBOX_NAME")
    DEPLOY_METHOD="toolbox ($TOOLBOX_NAME)"
else
    DEPLOYER="$SCRIPT_DIR/deploy_rocm_host.sh"
    EXTRA_ARGS=()
    DEPLOY_METHOD="host"
fi

# CPU deployment function for models without ROCm support
deploy_cpu_model() {
    local path="$1"
    local label="$2"

    log_info "Deploying $label using CPU deployment..."

    # Find the correct function config file
    local func_config=""
    if [[ -f "$path/function.yaml" ]]; then
        func_config="$path/function.yaml"
    elif [[ -f "$path/nuclio/function.yaml" ]]; then
        func_config="$path/nuclio/function.yaml"
        path="$path/nuclio"  # Update path to nuclio directory
    else
        log_error "Could not find function.yaml in $path or $path/nuclio/"
        return 1
    fi

    log_info "Using function config: $func_config"

    # Build custom base image if Dockerfile exists
    if [[ -f "$path/Dockerfile" ]]; then
        local image_tag="cvat.$(basename "$(dirname "$path")").$(basename "$path").cpu"
        log_info "Building custom base image: $image_tag"
        docker build -t "$image_tag" "$path" || {
            log_warning "Failed to build custom image, continuing with default..."
        }
    fi

    # Deploy using nuctl directly
    nuctl deploy --project-name cvat \
        --path "$path" \
        --file "$func_config" \
        --platform local \
        --env CVAT_FUNCTIONS_REDIS_HOST=cvat_redis_ondisk \
        --env CVAT_FUNCTIONS_REDIS_PORT=6666 \
        --platform-config '{"attributes": {"network": "cvat_cvat"}}'
}

# Check if deployer script exists and is executable
if [[ ! -x "$DEPLOYER" ]]; then
    log_error "Deployer script not found or not executable: $DEPLOYER"
    log_error "Make sure the ROCm deployment scripts are available in $SCRIPT_DIR"
    exit 1
fi

# Function to deploy a model
deploy_model() {
    local label="$1"
    local path="$2"
    local deployment_type="$3"

    log_info "Deploying $label ($deployment_type)..."

    if [[ ! -d "$path" ]]; then
        log_warning "Skipping $label (path not found: $path)"
        return 0
    fi

    # Check for appropriate config files
    local has_config=false
    local config_file=""

    if [[ "$deployment_type" == "ROCm" ]]; then
        config_file="$path/nuclio/function-rocm.yaml"
        # Fallback to function-gpu.yaml if ROCm not available
        if [[ ! -f "$config_file" ]]; then
            config_file="$path/nuclio/function-gpu.yaml"
            if [[ -f "$config_file" ]]; then
                log_info "$label: Using GPU config as ROCm fallback"
            fi
        fi
    else
        config_file="$path/nuclio/function.yaml"
    fi

    if [[ -f "$config_file" ]]; then
        has_config=true
    fi

    if [[ "$has_config" != true ]]; then
        log_warning "Skipping $label (no appropriate config file found)"
        log_warning "Looking for: $config_file"
        return 0
    fi

    log_info "Using config file: $config_file"

    # Choose the appropriate deployment method
    if [[ "$deployment_type" == "CPU" ]]; then
        log_info "Using CPU deployment for $label"
        # Run CPU deployment in subshell to prevent script exit on failure
        if (
            set +e  # Disable exit on error for this subshell
            deploy_cpu_model "$path" "$label"
            exit_code=$?
            exit $exit_code
        ); then
            log_success "Successfully deployed $label"
        else
            log_error "Failed to deploy $label"
            return 1
        fi
    else
        # ROCm deployment
        local actual_args=("${EXTRA_ARGS[@]}")
        if [[ "$USE_TOOLBOX" != true && ! -f "$path/nuclio/function-rocm.yaml" && -f "$path/nuclio/function-gpu.yaml" ]]; then
            # Use ROCm deployer with GPU fallback for GPU-only configs
            actual_args+=("--env" "INCLUDE_GPU_FALLBACK=1")
            log_info "Using GPU fallback for $label (ROCm config not available)"
        fi

        # Run ROCm deployment in subshell to prevent script exit on failure
        if (
            set +e  # Disable exit on error for this subshell
            "$DEPLOYER" "$path" "${actual_args[@]}"
            exit_code=$?
            exit $exit_code
        ); then
            log_success "Successfully deployed $label"
        else
            log_error "Failed to deploy $label"
            return 1
        fi
    fi
}

# Main deployment logic
log_info "Starting egocentric model deployment"
log_info "Deployment method: $DEPLOY_METHOD"
log_info "ROCm acceleration: $( [[ "$USE_ROCM" = true ]] && echo "enabled" || echo "disabled (using CPU)" )"

# Determine which models to deploy
if [[ "$DEPLOY_ALL" = true ]]; then
    DEPLOY_SAM=true
    DEPLOY_DETECTRON2=true
    DEPLOY_MMPOSE=true
    DEPLOY_MEDIAPIPE=true
fi

# Deploy models
deployed_count=0
failed_count=0

log_info "Starting individual model deployments..."

# SAM - Interactive Segmentation
if [[ "$DEPLOY_SAM" = true ]]; then
    log_info "Processing SAM deployment..."
    if deploy_model "SAM (Interactive Segmentation)" "$SCRIPT_DIR/pytorch/facebookresearch/sam" "ROCm"; then
        ((deployed_count++))
        log_info "SAM deployment completed successfully"
    else
        ((failed_count++))
        log_error "SAM deployment failed"
    fi
    log_info "SAM processing block completed, moving to next model..."
fi

# SAM Auto - Automatic Segmentation
if [[ "$DEPLOY_SAM_AUTO" = true ]]; then
    log_info "Processing SAM Auto deployment..."
    if deploy_model "SAM Auto (Automatic Segmentation)" "$SCRIPT_DIR/pytorch/facebookresearch/sam" "ROCm" "function-detector.yaml"; then
        ((deployed_count++))
        log_info "SAM Auto deployment completed successfully"
    else
        ((failed_count++))
        log_error "SAM Auto deployment failed"
    fi
    log_info "SAM Auto processing block completed, moving to next model..."
fi

# Detectron2 RetinaNet - Instance Segmentation
if [[ "$DEPLOY_DETECTRON2" = true ]]; then
    log_info "Processing Detectron2 deployment..."
    if deploy_model "Detectron2 RetinaNet R101 (Instance Segmentation)" "$SCRIPT_DIR/pytorch/facebookresearch/detectron2/retinanet_r101" "ROCm"; then
        ((deployed_count++))
        log_info "Detectron2 deployment completed successfully"
    else
        ((failed_count++))
        log_error "Detectron2 deployment failed"
    fi
fi

# MMPose - Hand Pose Estimation
if [[ "$DEPLOY_MMPOSE" = true ]]; then
    log_info "Processing MMPose deployment..."
    # Check if ROCm config exists, otherwise fall back to CPU
    mmpose_path="$SCRIPT_DIR/pytorch/mmpose/hrnet32"
    if [[ "$USE_ROCM" = true ]] && [[ -f "$mmpose_path/nuclio/function-rocm.yaml" ]]; then
        deployment_type="ROCm"
    else
        deployment_type="CPU"
        log_info "MMPose: Falling back to CPU deployment (ROCm not available)"
    fi

    if deploy_model "MMPose HRNet-W32 (Hand Pose Estimation)" "$mmpose_path" "$deployment_type"; then
        ((deployed_count++))
        log_info "MMPose deployment completed successfully"
    else
        ((failed_count++))
        log_error "MMPose deployment failed"
    fi
fi

# MediaPipe - Lightweight Pose Estimation
if [[ "$DEPLOY_MEDIAPIPE" = true ]]; then
    log_info "Processing MediaPipe deployment..."
    # MediaPipe works well on both CPU and ROCm
    mediapipe_path="$SCRIPT_DIR/pytorch/google/mediapipe-pose"
    if [[ "$USE_ROCM" = true ]] && [[ -f "$mediapipe_path/nuclio/function-rocm.yaml" ]]; then
        deployment_type="ROCm"
    else
        deployment_type="CPU"
        log_info "MediaPipe: Using CPU deployment (ROCm config not available)"
    fi

    if deploy_model "MediaPipe Pose (33 keypoints)" "$mediapipe_path" "$deployment_type"; then
        ((deployed_count++))
        log_info "MediaPipe deployment completed successfully"
    else
        ((failed_count++))
        log_error "MediaPipe deployment failed"
    fi
fi

log_info "All model deployments processed"

# Summary
echo
log_info "Deployment Summary:"
log_info "  Models deployed successfully: $deployed_count"
if [[ $failed_count -gt 0 ]]; then
    log_warning "  Models failed to deploy: $failed_count"
fi

if [[ $deployed_count -gt 0 ]]; then
    log_success "Egocentric model deployment completed!"
    echo
    log_info "Next steps:"
    log_info "  1. Check Nuclio dashboard at http://localhost:8070"
    log_info "  2. Test the deployed functions with egocentric dataset samples"
    log_info "  3. Use CVAT's auto-annotation feature with the deployed models"
else
    log_error "No models were successfully deployed. Check the logs above for details."
    exit 1
fi
