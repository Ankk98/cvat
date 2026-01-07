#!/bin/bash
# Deploy Nuclio functions through an AMD ROCm-enabled toolbox container.
#
# Usage:
#   ./deploy_rocm_toolbox.sh [functions_dir] [toolbox_name]
# Environment overrides:
#   ROCM_TOOLBOX_NAME   - toolbox container name (default: strix-halo-llm-finetuning)
#   NUCTL_BIN           - nuctl binary to invoke (default: nuctl)
#   NUCTL_PLATFORM      - Nuclio platform target (default: local)
#   NUCLIO_NETWORK      - Docker network name for CVAT stack (default: cvat_cvat)
#   NUCLIO_EXTRA_ARGS   - Extra flags forwarded to nuctl deploy
#   INCLUDE_GPU_FALLBACK- Set to 1 to deploy function-gpu.yaml when no ROCm file exists
#
# The script scans for files named function-rocm.yaml (preferred) or
# function-gpu.yaml under the provided directory and deploys each one
# through the selected toolbox so that ROCm, RADV, and other GPU
# dependencies stay isolated inside the toolbox image.

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
FUNCTIONS_DIR="${1:-$SCRIPT_DIR}"
DEFAULT_TOOLBOX_NAME="strix-halo-llm-finetuning"
TOOLBOX_NAME="${2:-${ROCM_TOOLBOX_NAME:-$DEFAULT_TOOLBOX_NAME}}"
NUCTL_BIN="${NUCTL_BIN:-nuctl}"
NUCTL_PLATFORM="${NUCTL_PLATFORM:-local}"
NUCLIO_NETWORK="${NUCLIO_NETWORK:-cvat_cvat}"
NUCLIO_EXTRA_ARGS="${NUCLIO_EXTRA_ARGS:-}"
INCLUDE_GPU_FALLBACK="${INCLUDE_GPU_FALLBACK:-0}"

if ! command -v toolbox >/dev/null 2>&1; then
    echo "toolbox command not found. Install Toolbox or update PATH." >&2
    exit 1
fi

if ! toolbox list | grep -q "$TOOLBOX_NAME"; then
    echo "Toolbox container '$TOOLBOX_NAME' not found. Create it before deploying." >&2
    exit 1
fi

run_in_toolbox() {
    local cmd=("$@")
    local quoted=""
    local arg

    for arg in "${cmd[@]}"; do
        quoted+=" $(printf '%q' "$arg")"
    done

    toolbox enter "$TOOLBOX_NAME" -- /bin/bash -lc "${quoted:1}"
}

run_in_toolbox "$NUCTL_BIN" create project cvat --platform "$NUCTL_PLATFORM" >/dev/null 2>&1 || true

shopt -s globstar nullglob
declare -A selected_configs=()

for path in "$FUNCTIONS_DIR"/**/function-rocm.yaml; do
    [ -f "$path" ] || continue
    func_root="$(dirname "$path")"
    selected_configs["$func_root"]="$path"
done

if [ "$INCLUDE_GPU_FALLBACK" = "1" ]; then
    for path in "$FUNCTIONS_DIR"/**/function-gpu.yaml; do
        [ -f "$path" ] || continue
        func_root="$(dirname "$path")"
        if [ -z "${selected_configs[$func_root]:-}" ]; then
            selected_configs["$func_root"]="$path"
        fi
    done
fi

if [ ${#selected_configs[@]} -eq 0 ]; then
    echo "No function-rocm.yaml or function-gpu.yaml files found under $FUNCTIONS_DIR" >&2
    exit 1
fi

mapfile -t sorted_roots < <(printf "%s\n" "${!selected_configs[@]}" | sort)
for root in "${sorted_roots[@]}"; do
    func_config="${selected_configs[$root]}"
    func_root="$(dirname "$func_config")"
    func_rel_path="$(realpath --relative-to="$SCRIPT_DIR" "$func_root")"

    echo "Deploying $func_rel_path via toolbox $TOOLBOX_NAME..."
    run_in_toolbox "$NUCTL_BIN" deploy \
        --project-name cvat \
        --path "$func_root" \
        --file "$func_config" \
        --platform "$NUCTL_PLATFORM" \
        --env CVAT_FUNCTIONS_REDIS_HOST=cvat_redis_ondisk \
        --env CVAT_FUNCTIONS_REDIS_PORT=6666 \
        --platform-config "{\"attributes\": {\"network\": \"$NUCLIO_NETWORK\"}}" \
        $NUCLIO_EXTRA_ARGS
done

run_in_toolbox "$NUCTL_BIN" get function --platform "$NUCTL_PLATFORM"

