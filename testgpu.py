import torch, platform
print("torch:", torch.__version__)
print("cuda_is_available:", torch.cuda.is_available())
print("build_cuda:", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
