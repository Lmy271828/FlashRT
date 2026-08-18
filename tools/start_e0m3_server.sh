#!/usr/bin/env bash
# Omega-QVLA x FlashRT E0M3 server on Thor.
#
# Lives in the repo (was ~/start_e0m3_server.sh) so it is version-controlled
# and rsyncs with the rest of tools/. Paths are derived from the script's
# own location; override via env if the Thor layout differs:
#
#   OPENPI_ROOT   default: repo root derived from this file (third_party/flashrt)
#   OMEGA_ROOT    default: $HOME/lmy/Omega-QVLA
#   OPENPI_CACHE  default: $HOME/.cache/openpi
#
# Usage:
#   bash third_party/flashrt/tools/start_e0m3_server.sh
#   OMEGA_E0M3_CUDA_GRAPH=1 bash third_party/flashrt/tools/start_e0m3_server.sh
set -euo pipefail

OPENPI_ROOT="${OPENPI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
OMEGA_ROOT="${OMEGA_ROOT:-$HOME/lmy/Omega-QVLA}"
OPENPI_CACHE="${OPENPI_CACHE:-$HOME/.cache/openpi}"

sudo docker stop pi05_server 2>/dev/null || true
sudo docker rm   pi05_server 2>/dev/null || true

sudo docker run -d --name pi05_server --runtime nvidia \
  --cap-add SYS_ADMIN --network host \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$OPENPI_ROOT":/workspace \
  -v "$OMEGA_ROOT":/opt/omega \
  -v "$OPENPI_CACHE":/root/.cache/openpi \
  -w /workspace \
  openpi-pi0.5:l4t-jp7.2 \
  bash -c "TF_DIR=/usr/local/lib/python3.12/dist-packages/transformers && \
           cp -r src/openpi/models_pytorch/transformers_replace/* \$TF_DIR/ && \
           export PYTHONPATH=packages/openpi-client/src:src:.:/opt/omega:/workspace/third_party/flashrt && \
           export OMEGA_E0M3_PACK=/workspace/third_party/flashrt/pi05_long_e0m3.pt && \
           export OMEGA_E0M3_CUDA_GRAPH=${OMEGA_E0M3_CUDA_GRAPH:-1} && \
           export GR00T_GPTQ=1 \
                  GR00T_GPTQ_PATH=/opt/omega/packs_hf/pi05_long/quantized.pt \
                  GR00T_GPTQ_INCLUDE='.*paligemma_with_expert\.gemma_expert\.model\.layers\.[0-9]+\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj).*' \
                  GR00T_GPTQ_WBITS_DEFAULT=4 GR00T_GPTQ_ABITS=4 GR00T_GPTQ_MISSING=fallback \
                  GR00T_DUQUANT_INCLUDE='.*paligemma_with_expert\.paligemma\.model\.language_model\.layers\.[0-9]+\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj).*' \
                  GR00T_DUQUANT_ROT_MODE=svd_hadamard GR00T_DUQUANT_PERM_SCORE=weight \
                  GR00T_DUQUANT_BLOCK=64 GR00T_DUQUANT_BLOCK_OUT=64 \
                  GR00T_DUQUANT_PERMUTE=1 GR00T_DUQUANT_ROW_ROT=restore \
                  GR00T_DUQUANT_ACT_PCT=99.9 GR00T_DUQUANT_CALIB_STEPS=32 GR00T_DUQUANT_LS=0.15 && \
           python -u /workspace/third_party/flashrt/tools/serve_omega_e0m3.py \
             --model_path /root/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
             --data_config pi05_libero --port 8000"

echo "pi05_server started; logs: sudo docker logs -f pi05_server"
