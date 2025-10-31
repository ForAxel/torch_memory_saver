import logging
import os
import shutil
from pathlib import Path
import setuptools
from setuptools import setup
from setuptools.command.build_ext import build_ext

from torch_musa.utils.musa_extension import BuildExtension

logger = logging.getLogger(__name__)


# copy & modify from torch/utils/cpp_extension.py
def _find_platform_home(platform):
    """Find the install path for the specified platform (cuda/rocm)."""
    if platform == "cuda" or platform == "musa":
        # Find CUDA home
        home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
        if home is None:
            compiler_path = shutil.which("nvcc")
            if compiler_path is not None:
                home = os.path.dirname(os.path.dirname(compiler_path))
            else:
                # home = '/usr/local/cuda'
                home = '/usr/local/musa'
    else:  # rocm/hip
        # Find ROCm home
        home = os.environ.get('ROCM_HOME') or os.environ.get('ROCM_PATH')
        if home is None:
            compiler_path = shutil.which("hipcc")
            if compiler_path is not None:
                home = os.path.dirname(os.path.dirname(compiler_path))
            else:
                home = '/opt/rocm'
    return home


def _detect_platform():
    """Detect whether to use CUDA or HIP based on available tools."""
    # Check for HIP first (since it might be preferred on AMD systems)
    if shutil.which("hipcc") is not None:
        return "hip"
    elif shutil.which("nvcc") is not None:
        return "cuda"
    else:
        # Default to CUDA if neither is found
        return "cuda"


class PlatformExtension(setuptools.Extension):
    """Unified extension class for both CUDA and HIP platforms."""
    def __init__(self, name, sources, platform="cuda", *args, **kwargs):
        self.platform = platform
        super().__init__(name, sources, *args, **kwargs)


# def _create_ext_modules(platform):
#     """Create extension modules based on the specified platform."""
    
#     # Common sources for all extensions
#     sources = [
#         'csrc/api_forwarder.cpp',
#         'csrc/core.cpp',
#         'csrc/entrypoint.cpp',
#     ]
    
#     # Common define macros
#     common_macros = [('Py_LIMITED_API', '0x03090000')]

#     # Common compile arguments
#     extra_compile_args = ['-std=c++17', '-O3']
    
#     # Platform-specific configurations
#     platform_home = Path(_find_platform_home(platform))
    
#     if platform == "hip":
#         # Add ROCm-specific source file
#         sources.append('csrc/hardware_amd_support.cpp')
        
#         include_dirs = [str(platform_home.resolve() / 'include')]
#         library_dirs = [str(platform_home.resolve() / 'lib')]
#         libraries = ['amdhip64', 'dl']
#         platform_macros = [('USE_ROCM', '1')]
#     else:  # cuda
#         include_dirs = [str((platform_home / 'include').resolve())]
#         library_dirs = [
#             str((platform_home / 'lib64').resolve()),
#             str((platform_home / 'lib64/stubs').resolve()),
#         ]
#         libraries = ['cuda', 'cudart']
#         platform_macros = [('USE_CUDA', '1')]
    
#     # Create extensions with different hook modes
#     ext_modules = [
#         PlatformExtension(
#             name,
#             sources,
#             platform=platform,
#             include_dirs=include_dirs,
#             library_dirs=library_dirs,
#             libraries=libraries,
#             define_macros=[
#                 *common_macros,
#                 *platform_macros,
#                 *extra_macros,
#             ],
#             py_limited_api=True,
#             extra_compile_args=extra_compile_args,
#         )
#         for name, extra_macros in [
#             ('torch_memory_saver_hook_mode_preload', [('TMS_HOOK_MODE_PRELOAD', '1')]),
#             ('torch_memory_saver_hook_mode_torch', [('TMS_HOOK_MODE_TORCH', '1')]),
#         ]
#     ]
    
#     return ext_modules


def _create_ext_modules_musa(platform) -> setuptools.Extension:
    """
    platform : string default cuda
    Setup MUSA extension for PyTorch support
    csrc_source_files, ./csrc
    csrc_header_files, ./csrc
    common_header_files, ?
    """

    # Source files
    sources = [
        'csrc/api_forwarder.cpp',
        'csrc/core.cpp',
        'csrc/entrypoint.cpp',
    ]

    import torch, torch_musa
    torch_musa_dir = Path(torch_musa.__file__).parent

    # Header files
    include_dirs = [
        torch_musa_dir / "share" / "torch_musa_codegen",
        "/home/torch_musa", # some *.muh not installed!
    ]

    # Common define macros
    common_macros = [('Py_LIMITED_API', '0x03090000')]

    # Common compile arguments
    # extra_compile_args = ['-std=c++17', '-O3']
    cxx_flags = [
        "-O3",
        "-fvisibility=hidden",
        "-std=c++17",
        "-Wno-reorder",
        "-march=native",
        "force_mcc",
    ]
    mcc_flags = [
        "-O3",
        "-march=native",
    ]
    
    # Platform-specific configurations
    platform_home = Path(_find_platform_home(platform)) # '/usr/local/musa'

    if platform=="cuda":
        # only support platform==cuda
        additional_include_dirs = [str((platform_home / 'include').resolve())] # ['/usr/local/musa-4.3.0/include']
        include_dirs.extend(additional_include_dirs) # TODO 
        library_dirs = [
            str((platform_home / 'lib64').resolve()),
            str((platform_home / 'lib64/stubs').resolve()),
        ] # ['/usr/local/musa-4.3.0/lib64', '/usr/local/musa-4.3.0/lib64/stubs']
        # libraries = ['cuda', 'cudart'] # TODO
        libraries = ['musa', 'musart']
        platform_macros = [('USE_CUDA', '1')] # TODO 保持传参不变?
        # platform_macros = [('USE_MUSA', '1')]
    else:
        raise NotImplementedError

    # # Libraries TODO 可能要替换
    # library_dirs = [
    #     torch_musa_dir / "lib"
    # ]
    # libraries = [
    #     "musa_kernels",
    #     "musa_python",
    # ]


    # Construct PyTorch CUDA extension
    from torch_musa.utils.musa_extension import MUSAExtension

    ext_modules = [
        MUSAExtension(
            name,
            sources,
            platform=platform, # 'musa'
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            define_macros=[
                *common_macros,
                *platform_macros,
                *extra_macros,
            ],
            py_limited_api=True,
            # extra_compile_args=extra_compile_args,
            extra_compile_args={
                "cxx": cxx_flags,
                "mcc": mcc_flags,
            },
        )
        for name, extra_macros in [
            ('torch_memory_saver_hook_mode_preload', [('TMS_HOOK_MODE_PRELOAD', '1')]),
            ('torch_memory_saver_hook_mode_torch', [('TMS_HOOK_MODE_TORCH', '1')]),
        ]
    ]
    
    return ext_modules


# Detect platform and set up accordingly
platform = _detect_platform()
print(f"Detected platform: {platform}")

# Create extension modules using unified function
# ext_modules = _create_ext_modules(platform)
ext_modules = _create_ext_modules_musa(platform) # 使用 MUSA 适配

setup(
    name='torch_memory_saver',
    version='0.0.9',
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension} ,
    # cmdclass={
    #     'build_ext': lambda: __import__(
    #         'torch_musa.utils.musa_extension', 
    #         fromlist=['BuildExtension']
    #     ).BuildExtension
    # },
    python_requires=">=3.9",
    packages=setuptools.find_packages(include=["torch_memory_saver", "torch_memory_saver.*"]),
)
