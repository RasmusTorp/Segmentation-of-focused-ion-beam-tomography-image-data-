import torchvision.transforms as v2

def get_transforms(gaussian_kernel_size, gaussian_sigma, brightness, contrast):
    transforms = []
        
    if gaussian_kernel_size:
        transforms.append(v2.GaussianBlur(kernel_size=gaussian_kernel_size, sigma = (0.001, gaussian_sigma)))
        
    if brightness or contrast:
        transforms.append(v2.ColorJitter(brightness=brightness, contrast=contrast))
    
    if not transforms:
        return None
    
    return v2.Compose(transforms)

def get_normalizer(X):
    mu = X.mean(axis=(0, 2, 3)).tolist()
    std = X.std(axis=(0, 2, 3)).tolist()
    return v2.Normalize(mean=mu, std=std)