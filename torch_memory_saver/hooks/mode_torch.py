import logging

from torch_memory_saver.hooks.base import HookUtilBase
from torch_memory_saver.utils import get_binary_path_from_package
import torch

def get_pluggable_allocator():
    # CUDA
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        from torch.cuda.memory import CUDAPluggableAllocator
        return CUDAPluggableAllocator

    # MUSA
    if hasattr(torch, "musa") and torch.musa.is_available():
        return torch.musa.memory.MUSAPluggableAllocator

    raise RuntimeError("No supported backend (CUDA/MUSA) available for pluggable allocator.")

PluggableAllocator = get_pluggable_allocator()

logger = logging.getLogger(__name__)


class HookUtilModeTorch(HookUtilBase):
    def __init__(self):
        self.allocator = PluggableAllocator(self.get_path_binary(), "tms_torch_malloc", "tms_torch_free")
        logger.debug(f"HookUtilModeTorch {self.allocator=} {self.get_path_binary()=}")

    def get_path_binary(self):
        return str(get_binary_path_from_package("torch_memory_saver_hook_mode_torch"))

    def get_allocator(self):
        return self.allocator.allocator()