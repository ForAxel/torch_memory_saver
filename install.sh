rm -rf build dist *.egg-info ./*.so

export NVTE_FRAMEWORK=musa
# 设置 MUSA 路径（根据实际安装位置调整）
export MUSA_HOME=/usr/local/musa

# 设置 Python 路径（确保包含 torch_musa）
export PYTHONPATH=$(python -c "import torch_musa; import os; print(os.path.dirname(torch_musa.__file__))"):$PYTHONPATH

# 设置库路径
export LD_LIBRARY_PATH=$MUSA_HOME/lib64:$LD_LIBRARY_PATH


pip uninstall torch_memory_saver -y
pip install --no-build-isolation --no-cache-dir -e .
