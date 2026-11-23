"""Map safetensors read-only on Windows without Torch's storage mapping."""
import json
import math
import mmap
import warnings
from importlib.machinery import PathFinder
import sys


def load_checkpoint(module, path):
    header_limit = module._SAFETENSORS_MAX_HEADER_SIZE
    with open(path, 'rb') as file:
        header_size = int.from_bytes(file.read(8), 'little')
        if not 0 < header_size <= header_limit:
            raise ValueError(f'Invalid safetensors header size: {path}')
        data_start = 8 + header_size
        header = json.loads(file.read(header_size))
        data = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)

    tensors = {}
    torch = module.torch
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='The given buffer is not writable')
        for name, info in header.items():
            if name == '__metadata__':
                continue
            start, end = info['data_offsets']
            shape, dtype = info['shape'], module._TYPES[info['dtype']]
            if not 0 <= start <= end <= len(data) - data_start or end - start != math.prod(shape) * dtype.itemsize:
                raise ValueError(f'Invalid safetensors range for {name}: {path}')
            if start == end:
                tensors[name] = torch.empty(shape, dtype=dtype)
            else:
                tensors[name] = torch.frombuffer(data, dtype=dtype, count=math.prod(shape), offset=data_start + start).view(shape)
    return tensors, header.get('__metadata__', {})


class ReadonlySafetensors(PathFinder):
    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        if fullname != 'comfy.utils':
            return None
        spec = super().find_spec(fullname, path, target)
        if spec is None:
            return None
        execute = spec.loader.exec_module

        def exec_module(module):
            execute(module)
            original = module.load_torch_file

            def load(path, safe_load=False, device=None, return_metadata=False):
                if (not module.comfy.memory_management.aimdo_enabled
                        and path.lower().endswith(('.safetensors', '.sft'))
                        and (device is None or device.type == 'cpu')):
                    tensors, metadata = load_checkpoint(module, path)
                    return (tensors, metadata) if return_metadata else tensors
                return original(path, safe_load, device, return_metadata)

            module.load_torch_file = load
            sys.meta_path.remove(cls)

        spec.loader.exec_module = exec_module
        return spec
