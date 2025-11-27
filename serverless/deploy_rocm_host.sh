#!/bin/bash
# Deploy Nuclio ROCm-ready functions directly from the host without using a toolbox.
# Assumes that each function bundle already specifies a ROCm-capable base image
# (e.g. docker.io/rocm/pytorch:rocm7.1_ubuntu22.04_py3.10_pytorch_2.2.2).
#
# Usage:
#   ./deploy_rocm_host.sh [functions_dir]
# Environment overrides:
#   NUCTL_BIN        - nuctl binary to invoke (default: nuctl)
#   NUCTL_PLATFORM   - Nuclio platform target (default: local)
#   NUCLIO_PROJECT   - Nuclio project name (default: cvat)
#   NUCLIO_NETWORK   - Docker network name for CVAT stack (default: cvat_cvat)
#   NUCLIO_EXTRA_ARGS- Extra flags forwarded to nuctl deploy
#   INCLUDE_GPU_FALLBACK - Set to 1 to also deploy function-gpu.yaml files (default: 0)

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
FUNCTIONS_DIR="${1:-$SCRIPT_DIR}"
NUCTL_BIN="${NUCTL_BIN:-nuctl}"
NUCTL_PLATFORM="${NUCTL_PLATFORM:-local}"
NUCLIO_PROJECT="${NUCLIO_PROJECT:-cvat}"
NUCLIO_NETWORK="${NUCLIO_NETWORK:-cvat_cvat}"
NUCLIO_EXTRA_ARGS="${NUCLIO_EXTRA_ARGS:-}"
INCLUDE_GPU_FALLBACK="${INCLUDE_GPU_FALLBACK:-0}"

if ! command -v "$NUCTL_BIN" >/dev/null 2>&1; then
    echo "Cannot find '$NUCTL_BIN' in PATH. Install nuctl or set NUCTL_BIN." >&2
    exit 1
fi

"$NUCTL_BIN" create project "$NUCLIO_PROJECT" --platform "$NUCTL_PLATFORM" >/dev/null 2>&1 || true

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
        # Only register GPU configs if no ROCm variant was found.
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
config_paths=()
for root in "${sorted_roots[@]}"; do
    config_paths+=("${selected_configs[$root]}")
done

for func_config in "${config_paths[@]}"; do
    func_root="$(dirname "$func_config")"
    func_rel_path="$(realpath --relative-to="$SCRIPT_DIR" "$func_root")"

    echo "Deploying $func_rel_path..."
    "$NUCTL_BIN" deploy \
        --project-name "$NUCLIO_PROJECT" \
        --path "$func_root" \
        --file "$func_config" \
        --platform "$NUCTL_PLATFORM" \
        --env CVAT_FUNCTIONS_REDIS_HOST=cvat_redis_ondisk \
        --env CVAT_FUNCTIONS_REDIS_PORT=6666 \
        --platform-config "{\"attributes\": {\"network\": \"$NUCLIO_NETWORK\"}}" \
        $NUCLIO_EXTRA_ARGS
done

"$NUCTL_BIN" get function --platform "$NUCTL_PLATFORM"

