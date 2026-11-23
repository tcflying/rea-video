# Run upstream main.py so comfy_aimdo initializes before ComfyUI imports server.
import os
import runpy
import sys
from pathlib import Path

# COMFY_DIR points at the ComfyUI checkout; defaults to the machine this was built on.
BASE = Path(os.environ.get("COMFY_DIR") or "G:/comfyui")
ENGINE = Path(__file__).resolve().parent

# Optional CUDA 13 torch runtime. A venv with torch cu130 already works;
# REA_TORCH_DIR prepends a workspace-local copy instead of installing into it.
TORCH = Path(os.environ["REA_TORCH_DIR"]) if os.environ.get("REA_TORCH_DIR") else ENGINE / "torch-cu130"
sys.path[:0] = [str(BASE), str(ENGINE / "pydeps")]
if (TORCH / "torch" / "version.py").is_file():
    sys.path.insert(0, str(TORCH))

# Optional runtime shim for the PyAV empty-audio-frame failure in SaveVideo
# (node 7). It only matters for graphs that hand the encoder an EMPTY audio
# frame, i.e. director runs with timeline output.audioMode="mute".
# The faithful upstream graphs (no audioMode => the model generates native
# audio) do NOT need it, so the patch is OFF by default and enabled with
# HYPIT_PYA_PATCH=1. Applies in-process only; no shared file is modified.
# See engine/patches/pyav_audio_chunk.py for the root-cause traceback.
if os.environ.get("HYPIT_PYA_PATCH", "0") == "1":
    sys.path.insert(0, str(ENGINE / "patches"))
    import pyav_audio_chunk  # noqa: F401,E402  (installs itself on import)

if os.name == "nt":
    sys.path.insert(0, str(ENGINE / "patches"))
    from readonly_safetensors import ReadonlySafetensors

    sys.meta_path.insert(0, ReadonlySafetensors)

main = str(BASE / "main.py")
sys.argv[0] = main
runpy.run_path(main, run_name="__main__")
