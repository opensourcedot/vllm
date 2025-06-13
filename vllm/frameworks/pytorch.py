import torch
import torch.nn as nn
import torchair
import torch_npu

from vllm.frameworks import Framework

class PyTorchFramework(Framework):
    """PyTorch framework plugin for vLLM."""
    cuda = torch.cuda
    npu = torch.npu
    framework_name = "pytorch"
    nn = nn
    dtype = torch.dtype
    bfloat16 = torch.bfloat16
    float16 = torch.float16
    float32 = torch.float32
    float64 = torch.float64
    long = torch.long
    int = torch.int
    int8 = torch.int8
    int32 = torch.int32
    uint8 = torch.uint8
    int64 = torch.int64
    Tensor = torch.Tensor
    distributed = torch.distributed
    device = torch.device
    Size = torch.Size
    multiprocessing = torch.multiprocessing
    profiler = torch.profiler
    llm_datadist = torchair.llm_datadist

    def __init__(self):
        super().__init__(self)

    def cat(self, tensors, dim=0):
        return torch.cat(tensors, dim)

    def empty(self, size, dtype=None, device=None):
        return torch.empty(size, dtype, device)
    
    def empty_like(self, tensor):
        return torch.empty_like(tensor)
    
    def tensor(self, data, dtype=None, device=None):
        return torch.tensor(data, dtype=dtype, device=device)
    
    def frombuffer(self, buffer, dtype=None, device=None):
        return torch.frombuffer(buffer, dtype=dtype, device=device)
    
    def set_inductor_compile_threads(self, num_threads: int):
        torch._inductor.config.compile_threads = num_threads

    def inference_mode(self):
        return torch.inference_mode()

    def ByteTensor(self, data=None, device=None):
        if data is None:
            return torch.ByteTensor(device=device)
        else:
            return torch.ByteTensor(data, device=device)
        
    def zeros(self, size, dtype=None, device=None):
        return torch.zeros(size, dtype=dtype, device=device)

    def _npu_reshape_and_cache(*args, **kwargs):
        return torch_npu._npu_reshape_and_cache(args, kwargs)

    def scatter_update_(data, indices, updates, axis=0):
        return torch_npu.scatter_update_(axis, indices, updates)
    
    # def npu_format_cast