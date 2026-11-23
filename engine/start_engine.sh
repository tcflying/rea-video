#!/usr/bin/env bash
# Independent H3 engine for rea-video.
# Upstream ComfyUI main.py wrapper so comfy_aimdo.control.init() runs before
# comfy.model_management is imported.
#
# Environment:
#   COMFY_DIR          ComfyUI checkout            (default /g/comfyui)
#   COMFY_MODELS       model root                  (default $COMFY_DIR/models)
#   REA_ENGINE_PORT    HTTP port                   (default 1022)
#   REA_LISTEN         bind address                (default 127.0.0.1)
#   REA_RESERVE_VRAM   GiB held back from torch    (default 6)
#   REA_SPARK          1 = also load ComfyUI-Spark-H3 (default 0)
set -euo pipefail

COMFY="${COMFY_DIR:-/g/comfyui}"
MODELS="${COMFY_MODELS:-$COMFY/models}"
PORT="${REA_ENGINE_PORT:-1022}"
LISTEN="${REA_LISTEN:-127.0.0.1}"
RESERVE="${REA_RESERVE_VRAM:-6}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_WIN="$(cygpath -am "$ROOT")"
mkdir -p "$ROOT/logs"

NODES=(ComfyUI_MiniMaxH3_Director ComfyUI-VideoHelperSuite ComfyUI-KJNodes)
SPARK_ARGS=()
if [[ "${REA_SPARK:-0}" == 1 ]]; then
  NODES+=(ComfyUI-Spark-H3)
  SPARK_ARGS=(--extra-model-paths-config "$ROOT_WIN/spark/extra_model_paths.yaml")
fi

cd "$COMFY"
exec "$COMFY/venv/Scripts/python.exe" -I -B "$ROOT_WIN/launch_engine.py" \
  --listen "$LISTEN" --port "$PORT" \
  --disable-auto-launch \
  --disable-fast-disk \
  --disable-async-offload \
  --disable-pinned-memory \
  --reserve-vram "$RESERVE" \
  --disable-cuda-malloc \
  --use-pytorch-cross-attention \
  --disable-assets \
  --disable-partner-nodes \
  --disable-all-custom-nodes \
  --whitelist-custom-nodes "${NODES[@]}" \
  "${SPARK_ARGS[@]}" \
  --models-directory "$MODELS" \
  --input-directory "$ROOT_WIN/input" \
  --output-directory "$ROOT_WIN/output" \
  --user-directory "$ROOT_WIN/user" \
  --temp-directory "$ROOT_WIN/temp" \
  --verbose "$@"
