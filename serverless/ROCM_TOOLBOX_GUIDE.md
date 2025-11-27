## ROCm Toolbox GPU Deployments for CVAT

This guide shows how to run Nuclio serverless functions on the AMD Strix Halo iGPU (ROCm) while keeping CVAT’s official serverless workflow intact. It complements the [automatic annotation setup guide](https://docs.cvat.ai//docs/administration/advanced/installation_automatic_annotation/) and the ROCm toolbox images described in [kyuz0/amd-strix-halo-toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes).

### Prerequisites
- CVAT stack launched with `docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml up -d`
- `nuctl` installed on the host and pointing to the local Nuclio dashboard
- Fedora Toolbox (or distrobox) container pre-created from the ROCm toolbox image, e.g. `toolbox create --container strix-halo-llm-finetuning --image docker.io/library/amd-strix-halo-llm:latest`
- Repo mounted identically inside the toolbox (default toolbox mounts the host FS 1:1, so `/home/<user>/repos/cvat` is the same inside the container)

### Deploying ROCm Functions
1. **Start the ROCm toolbox shell if you want an interactive session**
   `toolbox enter strix-halo-llm-finetuning`
2. **Automated deployment**
   - Host-only deployment (preferred when functions already use ROCm PyTorch base images):
     ```bash
     ./serverless/deploy_rocm_host.sh serverless/pytorch
     ./serverless/deploy_rocm_robotics_models.sh
     ./serverless/deploy_rocm_pointcloud_models.sh
     ```
     (`deploy_rocm_host.sh` automatically prefers `function-rocm.yaml`; set `INCLUDE_GPU_FALLBACK=1` only if you need to deploy a `function-gpu.yaml` because no ROCm variant exists.)
   - Toolbox deployment (set `ROCM_USE_TOOLBOX=1` or pass a toolbox name if you still want the containerized workflow):
     ```bash
     ROCM_USE_TOOLBOX=1 ./serverless/deploy_rocm_robotics_models.sh strix-halo-llm-finetuning
     ROCM_USE_TOOLBOX=1 ./serverless/deploy_rocm_pointcloud_models.sh strix-halo-llm-finetuning
     ./serverless/deploy_rocm_toolbox.sh serverless/pytorch strix-halo-llm-finetuning
     ```
   - Uncomment the additional lines inside `deploy_rocm_robotics_models.sh` when you want to enable alternative models (YOLO, SiamMask, etc.).
3. **Verify the functions** via the Nuclio dashboard at `http://localhost:8070` or by running `toolbox enter strix-halo-llm-finetuning -- nuctl get function --platform local`.

### Switching Models Quickly
- The helper scripts automatically deploy `function-rocm.yaml` when present and only fall back to `function-gpu.yaml` if `INCLUDE_GPU_FALLBACK=1`. Add ROCm-specific configs per model for fine-grained control (e.g., ROCm base image, flash-attention flags, extra volumes) and keep legacy GPU configs around only as a backup.
- Keep multiple candidate models in the robotics script commented out so you can toggle them by editing a single file instead of handling long `nuctl` commands manually.

### Stopping, Freeing Resources, and Cleanup
- **Stop CVAT + serverless stack**
  `docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml down`
- **Remove a specific function**
  `toolbox enter strix-halo-llm-finetuning -- nuctl delete function <function-name> --platform local`
- **Wipe the entire Nuclio project (all functions)**
  `toolbox enter strix-halo-llm-finetuning -- nuctl delete project cvat --platform local --cascade`
- **Free toolbox/container resources**
  - Stop running toolbox session: `toolbox rm --force strix-halo-llm-finetuning`
  - Delete the huge ROCm image if needed: `podman rmi docker.io/library/amd-strix-halo-llm:latest`
- **Prune residual volumes/networks** (only if no other containers need them):
  `docker system prune` and `docker network rm cvat_cvat`

Re-run the helper scripts whenever you refresh the models or after pruning the Nuclio project.

