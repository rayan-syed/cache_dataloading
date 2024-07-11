import torch

def cuda_available():
    return torch.cuda.is_available()

if __name__ == "__main__":
    if cuda_available():
        print("CUDA is available on this system.")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDNN version: {torch.backends.cudnn.version()}")
        print(f"GPU count: {torch.cuda.device_count()}")
        current_device = torch.cuda.current_device()
        print(f"Current CUDA device: {torch.cuda.get_device_name(current_device)}")
    else:
        print("CUDA is not available on this system.")
