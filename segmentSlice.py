import torch
from torch.utils.data import TensorDataset, DataLoader
from itertools import product

def segment_slice(model, X, sampling_height, sampling_width):
    height, width = X.shape
    
    n_space_width = width // sampling_width
    n_space_height = height // sampling_height

    x_crops = [i * sampling_width for i in range(n_space_width)]
    y_crops = [i * sampling_height for i in range(n_space_height)]
    
    
    # List of (x, y) pairs upper left corner
    crop_pairs = list(product(x_crops, y_crops))
    
    # Crop the images
    X_list = [X[:, :, y_:y_+sampling_height, x_:x_+sampling_width] for x_, y_ in crop_pairs]

    # If y is your labels tensor and you want to crop it in the same way
    y_list = [y[:, y_:y_+sampling_height, x_:x_+sampling_width] for x_, y_ in crop_pairs]
    
    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)