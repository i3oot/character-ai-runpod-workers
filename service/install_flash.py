"""Use the upstream release wheel matching this exact Python/Torch ABI."""
import subprocess
import sys
import torch

abi = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()
py = f'cp{sys.version_info.major}{sys.version_info.minor}'
wheel = f'flash_attn-2.8.3+cu12torch2.7cxx11abi{abi}-{py}-{py}-linux_x86_64.whl'
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps',
    'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/' + wheel], check=True)
