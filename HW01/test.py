import torch
print("CUDA可用:", torch.cuda.is_available())
print("GPU数量:", torch.cuda.device_count())
print("GPU名称:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
