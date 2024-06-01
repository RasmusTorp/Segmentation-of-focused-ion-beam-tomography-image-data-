import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from utils import get_normalizer
from sklearn.model_selection import train_test_split
from itertools import product

class InMemoryDataset(Dataset):
    def __init__(self, X, y, random_sampling = False, sampling_height = None, sampling_width = None, normalizer = None, dataAugmentations = None):
        
        if random_sampling:
            # Calculate the starting indices for the crops
            
            n_space_width = X.shape[3] // sampling_width
            n_space_height = X.shape[2] // sampling_height

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

        self.X = X
        self.y = y
        self.sampling_height = sampling_height
        self.sampling_width = sampling_width
        self.random_sampling = random_sampling
    
        self.dataAugmentations = dataAugmentations
        self.normalizer = normalizer
    
    def __len__(self):
        return len(self.y)
    
    #TODO: Flipping both X and y
    def __getitem__(self, idx):
        X_i, y_i = self.X[idx], self.y[idx]
        
        if self.dataAugmentations:
            X_i, y_i = self.dataAugmentations(X_i, y_i)
            
        if self.normalizer:
            X_i = self.normalizer(X_i)
        
        return X_i, y_i


def get_dataloaders(X, y, batch_size:int=15, train_size:float = 0.8, test_size:float = 0.2,seed:int = 42, 
                    sampling_height = None, sampling_width = None, static_test = False,
                    random_train_test_split=True, detector = "both", dataAugmentations = None):

    if detector == "1":
        X = X[:,0:1,:,:]
        
    elif detector == "2":
        X = X[:,1:2,:,:]
    if random_train_test_split:
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X.numpy(), y.numpy(), test_size=test_size, random_state=seed)
        
        train_size_int = int(train_size * len(X))
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=train_size_int, random_state=seed)
        
        # Convert back to PyTorch tensors
        X_train = torch.tensor(X_train)
        y_train = torch.tensor(y_train)
        X_test = torch.tensor(X_test)
        y_test = torch.tensor(y_test)
        X_val = torch.tensor(X_val)
        y_val = torch.tensor(y_val) 
        
    else:
        train_size = int(train_size * len(X))   
        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:], y[train_size:]
    
    normalizer = get_normalizer(X_train)
    
    train_dataset = InMemoryDataset(X_train, y_train, normalizer=normalizer, dataAugmentations = dataAugmentations)
    
    test_dataset = InMemoryDataset(X_test, y_test, sampling_height=sampling_height, sampling_width=sampling_width, 
                                    random_sampling = static_test,normalizer=normalizer)
    
    val_dataset = InMemoryDataset(X_val, y_val, sampling_height=sampling_height, sampling_width=sampling_width, 
                                    random_sampling = static_test, normalizer=normalizer)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False)

    return train_loader, test_loader, val_loader

