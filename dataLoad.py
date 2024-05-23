import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data.dataset import Dataset, random_split, TensorDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision.transforms import v2
from itertools import product

# class CustomDataset(Dataset):
#     def __init__(self, train_size = None, test_size = None, random_slicing = False, sampling_height = None, sampling_width = None,
#                 random_sampling = False, folder_path = "data/11t51center", transforms = None):
        
#         self.labels_filepath = folder_path + "/Segmented"
#         self.X_1_filepath = folder_path + "/Slicefront_corrected/Detector1"
#         self.X_2_filepath = folder_path + "/Slicefront_corrected/Detector2"

#         self.labels_filenames = sorted(os.listdir(self.labels_filepath))
#         self.X_1_filenames = sorted(os.listdir(self.X_1_filepath))
#         self.X_2_filenames = sorted(os.listdir(self.X_2_filepath))

#         self.X_1_filenames.pop(541)
#         self.X_2_filenames.pop(541)

#         self.sampling_height = sampling_height
#         self.sampling_width = sampling_width
#         self.random_sampling = random_sampling

#         if not random_slicing:
#             if train_size:
#                 self.X_1_filenames = self.X_1_filenames[:int(len(self.X_1_filenames) * train_size)]
#                 self.X_2_filenames = self.X_2_filenames[:int(len(self.X_2_filenames) * train_size)]
#                 self.labels_filenames = self.labels_filenames[:int(len(self.labels_filenames) * train_size)]

#             if test_size:
#                 train_size = 1 - test_size
#                 self.X_1_filenames = self.X_1_filenames[int(len(self.X_1_filenames) * train_size):]
#                 self.X_2_filenames = self.X_2_filenames[int(len(self.X_2_filenames) * train_size):]
#                 self.labels_filenames = self.labels_filenames[int(len(self.labels_filenames) * train_size):]

#         self.transforms = transforms
    
#     def __len__(self):
#         return len(self.labels_filenames)
    
#     def get_random_square(self, X, y):
#         h, w = X.shape[-2:]

#         start_h = np.random.randint(0, h - self.sampling_height)
#         start_w = np.random.randint(0, w - self.sampling_width)
        
#         X = X[:, start_h:start_h+self.sampling_height, start_w:start_w+self.sampling_width]
#         y = y[start_h:start_h+self.sampling_height, start_w:start_w+self.sampling_width]
        
#         return X, y

#     def __getitem__(self, idx):
#         X_1_path = os.path.join(self.X_1_filepath, self.X_1_filenames[idx])
#         X_2_path = os.path.join(self.X_2_filepath, self.X_2_filenames[idx])
#         label_path = os.path.join(self.labels_filepath, self.labels_filenames[idx])

#         X_1 = plt.imread(X_1_path)
#         X_2 = plt.imread(X_2_path)

#         X_1 = X_1[357-1:900, 4-1:900]
#         X_2 = X_2[357-1:900, 4-1:900]

#         X = np.stack([X_1,X_2], axis=0)

#         y = plt.imread(label_path)

#         X, y = torch.tensor(X), torch.tensor(y)

#         class_mapping = {0: 0, 128: 1, 255: 2}
#         mapped_labels = torch.zeros_like(y, dtype=torch.long)

#         for original_class, mapped_class in class_mapping.items():
#             mapped_labels[y == original_class] = mapped_class

#         y = mapped_labels

#         X = X.float()

#         if self.random_sampling:
#             X, y = self.get_random_square(X,y)
        
#         if self.transforms:
#             X = self.transforms(X)
        
#         return X, y

class InMemoryDataset(Dataset):
    def __init__(self, X, y, random_sampling = False, sampling_height = None, sampling_width = None, transforms = None, p_flip_horizontal = 0.0):
        
        if not random_sampling and sampling_height:
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
            # starts_h = [0, 0, X.shape[2] - square_size, X.shape[2] - square_size, 144, 144]
            # starts_w = [0, X.shape[3] - square_size, 0, X.shape[3] - square_size, 320, 641]

            # # Crop the images and labels
            # X_crops = [X[:, :, start_h:start_h+square_size, start_w:start_w+square_size] for start_h, start_w in zip(starts_h, starts_w)]
            # y_crops = [y[:, start_h:start_h+square_size, start_w:start_w+square_size] for start_h, start_w in zip(starts_h, starts_w)]

            # # Concatenate the crops along the batch dimension
            # X = np.concatenate(X_crops, axis=0)
            # y = np.concatenate(y_crops, axis=0)

        self.X = X
        self.y = y
        self.sampling_height = sampling_height
        self.sampling_width = sampling_width
        self.random_sampling = random_sampling
    
        self.transforms = transforms
        self.p_flip_horizontal = p_flip_horizontal
    
        # self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def __len__(self):
        return len(self.y)
    
    #TODO: Support non square sizes
    def get_random_square(self, X, y):
        h, w = X.shape[-2:]

        start_h = np.random.randint(0, h - self.sampling_height)
        start_w = np.random.randint(0, w - self.sampling_width)
        
        X = X[:, start_h:start_h+self.sampling_height, start_w:start_w+self.sampling_width]
        y = y[start_h:start_h+self.sampling_height, start_w:start_w+self.sampling_width]
        return X, y


    #TODO: Flipping both X and y
    def __getitem__(self, idx):
        flip_horizontal = self.p_flip_horizontal > np.random.rand()
        if self.sampling_height and self.random_sampling:    
            X_i, y_i = self.get_random_square(self.X[idx], self.y[idx])   
        
        else:
            X_i, y_i = self.X[idx], self.y[idx]
            
        if self.transforms:
            X_i = self.transforms(X_i)
        
        if flip_horizontal:
            x_channels = X_i.shape[0]
            X_i = torch.flip(X_i, [x_channels])  # Flip horizontally
            y_i = torch.flip(y_i, [1])  # Flip horizontally
            
        return X_i, y_i


def get_transforms(X, normalize = True):
    transforms = []
    
    if normalize:
        mu = X.mean(axis=(0, 2, 3)).tolist()
        std = X.std(axis=(0, 2, 3)).tolist()
        transforms.append(v2.Normalize(mean=mu, std=std))
        
    return v2.Compose(transforms)

def get_dataloaders(X, y, batch_size:int=15, train_size:float = 0.8, test_size:float = 0.2,seed:int = 42, verbose:bool = True, 
                    sampling_height = None, sampling_width = None, in_memory = False, static_test = False, random_sampling_train = False,
                    random_train_test_split=True, folder_path = "data/11t51center", detector = "both", normalize = True, p_flip_horizontal = 0.):

    
    
    # if in_memory:
    #     X, y = data_load_tensors(verbose, folder_path=folder_path)
        
    
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
        
        # transforms = get_transforms(X_train, normalize = normalize, p_flip_horizontal = p_flip_horizontal)
        transforms = get_transforms(X_train, normalize=normalize)
        
    else:
        train_size = int(train_size * len(X))   
        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:], y[train_size:]
    train_dataset = InMemoryDataset(X_train, y_train, sampling_height=sampling_height, sampling_width=sampling_width,random_sampling = random_sampling_train, transforms = transforms, p_flip_horizontal = p_flip_horizontal)
    test_dataset = InMemoryDataset(X_test, y_test, sampling_height=sampling_height, sampling_width=sampling_width, random_sampling = random_sampling_train, transforms = transforms)
    val_dataset = InMemoryDataset(X_val, y_val, sampling_height=sampling_height, sampling_width=sampling_width, random_sampling = static_test, transforms = transforms)
    
    # #TODO: random sampling for this too
    # else:
    #     train_dataset = CustomDataset(train_size = train_size, sampling_height=sampling_height, sampling_width=sampling_width, random_sampling = True, folder_path=folder_path, normalize = normalize, p_flip_horizontal = p_flip_horizontal)

    #     if static_test:
    #         X, y = data_load_tensors(verbose, folder_path=folder_path)
    #         train_size = int(train_size * len(X))   
    #         X_test, y_test = X[train_size:], y[train_size:]
    #         test_dataset = InMemoryDataset(X_test, y_test, sampling_height=sampling_height, sampling_width=sampling_width, random_sampling = False, normalize = normalize, p_flip_horizontal = p_flip_horizontal)

    #     else:
    #         test_dataset = CustomDataset(test_size = 1 - train_size, sampling_height=sampling_height, sampling_width=sampling_width, random_sampling = True, normalize = normalize, p_flip_horizontal = p_flip_horizontal)


    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False)

    return train_loader, test_loader, val_loader

