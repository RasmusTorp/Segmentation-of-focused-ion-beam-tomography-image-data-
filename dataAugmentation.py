import torchvision.transforms as v2
import torch
import numpy as np

class DataAugmentation:
    def __init__(self, gaussian_kernel_size, gaussian_sigma, brightness, contrast, p_flip_horizontal, sampling_width, sampling_height, random_sampling_train):
        transforms = []
        
        if gaussian_kernel_size:
            transforms.append(v2.GaussianBlur(kernel_size=gaussian_kernel_size, sigma = (0.001, gaussian_sigma)))
            
        if brightness or contrast:
            transforms.append(v2.ColorJitter(brightness=brightness, contrast=contrast))
        
        if not transforms:
            return None
        
        self.transforms = v2.Compose(transforms)
    
        self.p_flip_horizontal = p_flip_horizontal
        self.sample_width = sampling_width
        self.sampling_height = sampling_height
        self.random_sampling_train = random_sampling_train
        
    def get_random_square(self, X, y):
        h, w = X.shape[-2:]

        start_h = np.random.randint(0, h - self.sampling_height)
        start_w = np.random.randint(0, w - self.sampling_width)
        
        X = X[:, start_h:start_h+self.sampling_height, start_w:start_w+self.sampling_width]
        y = y[start_h:start_h+self.sampling_height, start_w:start_w+self.sampling_width]
        return X, y

    def flip_horizontal(self, X, y):
        x_channels = X_i.shape[0]
        X_i = torch.flip(X_i, [x_channels])  # Flip horizontally
        y_i = torch.flip(y_i, [1])  # Flip horizontally
        return X, y
    
    def __call__(self, X, y):
        if self.transforms:
            X = self.transforms(X)
        
        if self.random_sampling_train:
            X, y = self.get_random_square(X, y)
        
        if self.p_flip_horizontal > 0:
            if np.random.rand() < self.p_flip_horizontal:
                X, y = self.flip_horizontal(X, y)
        
        return X, y
