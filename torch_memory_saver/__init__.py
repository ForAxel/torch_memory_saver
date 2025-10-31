import musa_patch # ATTN musa patch 替换 cuda

from .entrypoint import TorchMemorySaver
from .hooks.mode_preload import configure_subprocess

# Global singleton
torch_memory_saver = TorchMemorySaver()
