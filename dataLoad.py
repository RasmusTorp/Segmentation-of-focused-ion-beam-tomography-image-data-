import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from torchvision.transforms import Normalize

from PIL import Image
class CustomDataset(Dataset):
    def __init__(self, train_size = None, test_size = None, square_size = None, random_sampling = False, normalize = True, folder_path = "data/11t51center"):
        self.labels_filepath = folder_path + "/Segmented"
        self.X_1_filepath = folder_path + "/Slicefront_corrected/Detector1"
        self.X_2_filepath = folder_path + "/Slicefront_corrected/Detector2"

        self.labels_filenames = sorted(os.listdir(self.labels_filepath))
        self.X_1_filenames = sorted(os.listdir(self.X_1_filepath))
        self.X_2_filenames = sorted(os.listdir(self.X_2_filepath))

        self.X_1_filenames.pop(541)
        self.X_2_filenames.pop(541)

        self.square_size = square_size 
        self.random_sampling = random_sampling

        if train_size:
            self.X_1_filenames = self.X_1_filenames[:int(len(self.X_1_filenames) * train_size)]
            self.X_2_filenames = self.X_2_filenames[:int(len(self.X_2_filenames) * train_size)]
            self.labels_filenames = self.labels_filenames[:int(len(self.labels_filenames) * train_size)]

        if test_size:
            train_size = 1 - test_size
            self.X_1_filenames = self.X_1_filenames[int(len(self.X_1_filenames) * train_size):]
            self.X_2_filenames = self.X_2_filenames[int(len(self.X_2_filenames) * train_size):]
            self.labels_filenames = self.labels_filenames[int(len(self.labels_filenames) * train_size):]

        if normalize:
            # Values calculated from the dataset
            self.normalize = Normalize(mean=[128.8504060979218, 91.33930754958642], std=[30.94350442869555, 27.551368693941175])

        else:
            self.normalize = False    
    
    def __len__(self):
        return len(self.labels_filenames)
    
    def get_random_square(self, X, y):
        h, w = X.shape[-2:]

        start_h = np.random.randint(0, h - self.square_size)
        start_w = np.random.randint(0, w - self.square_size)
        
        X = X[:, start_h:start_h+self.square_size, start_w:start_w+self.square_size]
        y = y[start_h:start_h+self.square_size, start_w:start_w+self.square_size]
        return X, y

    def __getitem__(self, idx):
        X_1_path = os.path.join(self.X_1_filepath, self.X_1_filenames[idx])
        X_2_path = os.path.join(self.X_2_filepath, self.X_2_filenames[idx])
        label_path = os.path.join(self.labels_filepath, self.labels_filenames[idx])

        X_1 = plt.imread(X_1_path)
        X_2 = plt.imread(X_2_path)

        X_1 = X_1[357-1:900, 4-1:900]
        X_2 = X_2[357-1:900, 4-1:900]

        X = np.stack([X_1,X_2], axis=0)

        y = plt.imread(label_path)

        X, y = torch.tensor(X), torch.tensor(y)

        class_mapping = {0: 0, 128: 1, 255: 2}
        mapped_labels = torch.zeros_like(y, dtype=torch.long)

        for original_class, mapped_class in class_mapping.items():
            mapped_labels[y == original_class] = mapped_class

        y = mapped_labels

        X = X.float()

        if self.random_sampling:
            X, y = self.get_random_square(X,y)
        
        return X, y

class InMemoryDataset(Dataset):
    def __init__(self, X, y, square_size = None, random_sampling = False):
        if not random_sampling and square_size:
            # Calculate the starting indices for the crops
            starts_h = [0, 0, X.shape[2] - square_size, X.shape[2] - square_size, 144, 144]
            starts_w = [0, X.shape[3] - square_size, 0, X.shape[3] - square_size, 320, 641]

            # Crop the images and labels
            X_crops = [X[:, :, start_h:start_h+square_size, start_w:start_w+square_size] for start_h, start_w in zip(starts_h, starts_w)]
            y_crops = [y[:, start_h:start_h+square_size, start_w:start_w+square_size] for start_h, start_w in zip(starts_h, starts_w)]

            # Concatenate the crops along the batch dimension
            X = np.concatenate(X_crops, axis=0)
            y = np.concatenate(y_crops, axis=0)

        self.X = X
        self.y = y
        self.square_size = square_size
        self.random_sampling = random_sampling
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def __len__(self):
        return len(self.y)
    
    def get_random_square(self, X, y):
        h, w = X.shape[-2:]

        start_h = np.random.randint(0, h - self.square_size)
        start_w = np.random.randint(0, w - self.square_size)
        
        X = X[:, start_h:start_h+self.square_size, start_w:start_w+self.square_size]
        y = y[start_h:start_h+self.square_size, start_w:start_w+self.square_size]
        return X, y

    def __getitem__(self, idx):
        if self.square_size and self.random_sampling:
            return self.get_random_square(self.X[idx], self.y[idx])
            
        return self.X[idx], self.y[idx]

def data_load_numpy(verbose:bool = True, processing = False, square_size = None, folder_path = "data/11t51center"):
    """
    Load and process image data from the specified folder path using numpy arrays.
    
    Args:
        verbose (bool): If True, print loading and processing updates.
        processing (bool): If True, process the loaded data.
        
    Returns:
        tuple: A tuple containing the image data (X) and corresponding labels.
    """

    labels_filepath = folder_path + "/Segmented"
    X_1_filepath = folder_path + "/Slicefront_corrected/Detector1"
    X_2_filepath = folder_path + "/Slicefront_corrected/Detector2"

    labels_filenames = sorted(os.listdir(labels_filepath))
    X_1_filenames = sorted(os.listdir(X_1_filepath))
    X_2_filenames = sorted(os.listdir(X_2_filepath))

    if verbose:
        print("\nLoading data...")

    labels = np.array([plt.imread(labels_filepath + "/" + filename) for filename in labels_filenames])
    X_1 = np.array([plt.imread(X_1_filepath + "/" + filename) for filename in X_1_filenames])
    X_2 = np.array([plt.imread(X_2_filepath + "/" + filename) for filename in X_2_filenames]) 

    # X_1 = np.array([np.squeeze(np.array(Image.open(X_1_filepath + "/" + filename).convert('L'))) for filename in X_1_filenames])
    # X_2 = np.array([np.squeeze(np.array(Image.open(X_2_filepath + "/" + filename).convert('L'))) for filename in X_2_filenames])
    # if verbose:
    #     print("\nData loaded.")

    if processing:
        if verbose:
            print("\nProcessing data...")


        X_1 = X_1[:,357-1:900, 4-1:900]
        X_2 = X_2[:,357-1:900, 4-1:900]

        frame_to_delete = 542
        X_1 = np.delete(X_1, frame_to_delete, axis=0)
        X_2 = np.delete(X_2, frame_to_delete, axis=0)

        if verbose:
            print("\nData processed.\n")
    
    X = np.stack([X_1,X_2], axis=0)
    
    X = np.transpose(X, (1, 0, 2, 3))

    if square_size:
        # Calculate the starting indices for the crops
        starts_h = [0, 0, X.shape[2] - square_size, X.shape[2] - square_size, 144, 144]
        starts_w = [0, X.shape[3] - square_size, 0, X.shape[3] - square_size, 320, 641]

        # Crop the images and labels
        X_crops = [X[:, :, start_h:start_h+square_size, start_w:start_w+square_size] for start_h, start_w in zip(starts_h, starts_w)]
        labels_crops = [labels[:, start_h:start_h+square_size, start_w:start_w+square_size] for start_h, start_w in zip(starts_h, starts_w)]

        # Concatenate the crops along the batch dimension
        X = np.concatenate(X_crops, axis=0)
        labels = np.concatenate(labels_crops, axis=0)

    return X, labels

def data_load_tensors(verbose:bool = True, processing:bool = True, one_hot:bool = False, map_lapels = True, square_size = None, folder_path = "data/11t51center"):
    """
    Load data into tensors and perform data processing.

    Args:
        verbose (bool): Whether to display verbose output.
        processing (bool): Whether to perform data processing.

    Returns:
        torch.Tensor: The input data in tensor format.
        torch.Tensor: The corresponding labels in tensor format.
    """
    X, y = data_load_numpy(verbose, processing, square_size=square_size, folder_path=folder_path)

    X, y = torch.tensor(X), torch.tensor(y)

    if map_lapels:
        class_mapping = {0: 0, 128: 1, 255: 2}
        mapped_labels = torch.zeros_like(y, dtype=torch.long)

        for original_class, mapped_class in class_mapping.items():
            mapped_labels[y == original_class] = mapped_class

        y = mapped_labels

        if one_hot:
            y = torch.nn.functional.one_hot(y, num_classes=3)
            y = y.permute(0, 3, 1, 2)

    # # Convert X to float
    X = X.float()
    return X, y 

def get_dataloaders(batch_size:int=15, train_size:float = 0.8, seed:int = 42, verbose:bool = True, square_size = None, in_memory = False,static_test = False, folder_path = "data/11t51center"):

    
    if in_memory:
        X, y = data_load_tensors(verbose, folder_path=folder_path)
        train_size = int(train_size * len(X))   
        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:], y[train_size:]
        train_dataset = InMemoryDataset(X_train, y_train, square_size=square_size, random_sampling = True)
        test_dataset = InMemoryDataset(X_test, y_test, square_size=square_size, random_sampling = False)

    else:
        train_dataset = CustomDataset(train_size = train_size, square_size=square_size, random_sampling = True, folder_path=folder_path)

        if static_test:
            X, y = data_load_tensors(verbose, folder_path=folder_path)
            train_size = int(train_size * len(X))   
            X_test, y_test = X[train_size:], y[train_size:]
            test_dataset = InMemoryDataset(X_test, y_test, square_size=square_size, random_sampling = False)

        else:
            test_dataset = CustomDataset(test_size = 1 - train_size, square_size=square_size, random_sampling = True)


    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=True)

    return train_loader, test_loader

if __name__ == "__main__":
    # (X_1, X_2), y = data_load_tensors()

    # X, y = data_load_tensors()

    # # Create a dataset
    # dataset = InMemoryDataset(X, y)
    train_size = 0.8
    test_size = 0.2

    trainSet = CustomDataset(train_size=train_size, square_size=256, random_sampling=True)
    testSet = CustomDataset(test_size=test_size, square_size=256, random_sampling=False)

    print("Done!")