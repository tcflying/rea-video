"""Borrow or start the local llama.cpp vision endpoint for the duration of a plan."""
from contextlib import contextmanager
import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


HOST = '127.0.0.1'
MODEL = 'qwen2.5-omni-7b'
# Override on other machines: llama.cpp build dir and the two GGUF files' dir.
BACKENDS = Path(os.environ.get('REA_VISION_BACKEND') or Path.home() / '.lmstudio/extensions/backends')
MODELS = Path(os.environ.get('REA_VISION_MODELS') or 'G:/comfyui/models/LLM/Qwen2.5-Omni-7B-GGUF')


def _get(port, path):
    # Direct connection: never follow a redirect or use a system proxy.
    conn = http.client.HTTPConnection(HOST, port, timeout=2)
    try:
        conn.request('GET', path)
        response = conn.getresponse()
        return json.loads(response.read(1 << 20)) if response.status == 200 else {}
    finally:
        conn.close()


def _ready(port, model):
    try:
        if _get(port, '/health').get('status') != 'ok':
            return False
        models = _get(port, '/v1/models')
        if not any(row.get('id') == model for row in models.get('data', [])):
            return False
        if any(row.get('name') == model and 'multimodal' in row.get('capabilities', [])
               for row in models.get('models', [])):
            return True
        return _get(port, '/props').get('modalities', {}).get('vision') is True
    except (OSError, http.client.HTTPException, ValueError, AttributeError, TypeError):
        return False


@contextmanager
def local_vision(port, model):
    """Yield an OpenAI base URL. Stop only a server launched by this context."""
    if not isinstance(port, int) or not 1 <= port <= 65535 or not isinstance(model, str) or not model.strip():
        raise ValueError('Expected a local TCP port and a model name')
    url = f'http://{HOST}:{port}/v1'
    if _ready(port, model):
        yield url
        return
    with socket.socket() as probe:
        if os.name == 'nt':
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            probe.bind((HOST, port))
        except OSError as exc:
            raise RuntimeError(f'{url} unavailable for startup; expected a healthy vision model {model}') from exc
    if model != MODEL:
        raise ValueError(f'Local startup supports {MODEL}; requested {model}')
    # REA_VISION_EXE / REA_VISION_VENDOR win; otherwise look for the LM Studio layout.
    exe = Path(os.environ['REA_VISION_EXE']) if os.environ.get('REA_VISION_EXE') else next(
        iter(sorted(BACKENDS.glob('llama.cpp-*/llama-server.exe'))), BACKENDS / 'llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.37.0/llama-server.exe')
    vendor = Path(os.environ['REA_VISION_VENDOR']) if os.environ.get('REA_VISION_VENDOR') else next(
        iter(sorted(BACKENDS.glob('vendor/win-llama-cuda12-vendor-*'))), BACKENDS / 'vendor/win-llama-cuda12-vendor-v2')
    weights = MODELS / 'Qwen2.5-Omni-7B-Q4_K_M.gguf'
    projector = MODELS / 'mmproj-Qwen2.5-Omni-7B-Q8_0.gguf'
    for path in (exe, vendor / 'cublas64_12.dll', weights, projector):
        if not path.is_file():
            raise FileNotFoundError(path)
    env = os.environ.copy()
    env['PATH'] = str(vendor) + os.pathsep + env.get('PATH', '')
    proc = subprocess.Popen([str(exe), '-m', str(weights), '--mmproj', str(projector),
                             '--host', HOST, '--port', str(port), '--alias', model,
                             '--offline', '--no-webui', '-c', '8192', '-np', '1',
                             '-ngl', '99', '--cache-ram', '0'], env=env, stdout=sys.stderr)
    try:
        deadline = time.monotonic() + 180
        while not _ready(port, model):
            if proc.poll() is not None:
                raise RuntimeError(f'llama-server exited with code {proc.returncode}')
            if time.monotonic() >= deadline:
                raise TimeoutError(f'{url} did not load {model} within 180 seconds')
            time.sleep(0.5)
        yield url
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


if __name__ == '__main__':
    with socket.socket() as probe:
        probe.bind((HOST, 0))
        port = probe.getsockname()[1]
    with local_vision(port, MODEL) as started:
        with local_vision(port, MODEL) as borrowed:
            assert started == borrowed
        assert _ready(port, MODEL), 'Borrower stopped the owner'
    assert not _ready(port, MODEL), 'Owner left its process running'
    print(f'Passed startup, reuse, cleanup on {HOST}:{port}')
