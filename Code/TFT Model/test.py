import torch, platform
print("torch:", torch.__version__)
print("cuda_is_available:", torch.cuda.is_available())
print("build_cuda:", getattr(torch.version, "cuda", None))
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
print("python:", platform.python_version())

import torch
print("cuda_is_available:", torch.cuda.is_available(),
      "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
