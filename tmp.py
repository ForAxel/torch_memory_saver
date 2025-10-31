import os
import shutil
import torch
import torch_musa


from build_tools.utils import get_frameworks

frameworks = get_frameworks()
print(f"frameworks: {frameworks}")


def _detect_platform():
    """Detect whether to use CUDA or HIP based on available tools."""
    # Check for HIP first (since it might be preferred on AMD systems)
    if shutil.which("hipcc") is not None:
        return "hip"
    elif shutil.which("nvcc") is not None:
        return "cuda"
    else:
        # Default to CUDA if neither is found
        return "musa"

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

def test_musa_embedding():
    # 测试基本的embedding操作
    embedding = torch.nn.Embedding(1000, 128).to("musa")
    input_ids = torch.tensor([[1, 2, 3, 4]], device="musa")
    output = embedding(input_ids)
    print("Embedding test passed")

if __name__ == "__main__":
    platform = _detect_platform()
    print(platform)
    print(_find_platform_home(platform))


    input_ids = torch.tensor([1,2,3,4])
    # 明确设置设备
    device = torch.device("musa")
    # 确保所有张量都在MUSA设备上
    input_ids = input_ids.to(device)
    print(f"input_ids: {input_ids}")

    test_musa_embedding()