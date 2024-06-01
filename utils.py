import torch
import torchvision.transforms as v2

def get_normalizer(X):
    mu = X.mean(axis=(0, 2, 3)).tolist()
    std = X.std(axis=(0, 2, 3)).tolist()
    return v2.Normalize(mean=mu, std=std)

def get_device():
    if torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")
    
    if torch.backends.mps.is_available():
        print("Using MPS")
        return torch.device("mps")
    
    print("Using CPU")
    return torch.device("cpu")