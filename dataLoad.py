import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import random_split
from torch.utils.data.dataset import TensorDataset, ConcatDataset, Dataset
from torch.utils.data import DataLoader

class CustomDataset(Dataset):
    def __init__(self):
        folder_path = "data/11t51center"

        self.labels_filepath = folder_path + "/Segmented"
        self.X_1_filepath = folder_path + "/Slicefront_corrected/Detector1"
        self.X_2_filepath = folder_path + "/Slicefront_corrected/Detector2"

        self.labels_filenames = os.listdir(self.labels_filepath)
        self.X_1_filenames = os.listdir(self.X_1_filepath)
        self.X_2_filenames = os.listdir(self.X_2_filepath)
    
    def __len__(self):
        return len(self.labels_filenames)
    
    def __getitem__(self, idx):
        X_1_path = os.path.join(self.X_1_filepath, self.X_1_filenames[idx])
        X_2_path = os.path.join(self.X_2_filepath, self.X_2_filenames[idx])
        label_path = os.path.join(self.labels_filepath, self.labels_filenames[idx])

        X_1 = plt.imread(X_1_path)
        X_2 = plt.imread(X_2_path)

        # Cropping
        X_1 = X_1[1-1:646,357-1:900, 4-1:900]
        X_2 = X_2[1-1:646,357-1:900, 4-1:900]
        
        label = plt.imread(label_path)
        
        return X_1, X_2, label


class InMemoryDataset(Dataset):
    def __init__(self, X, y):
        # X, y = data_load_tensors()
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def data_load_numpy(verbose:bool = True, processing:bool = True):
    """
    Load and process image data from the specified folder path using numpy arrays.
    
    Args:
        verbose (bool): If True, print loading and processing updates.
        processing (bool): If True, process the loaded data.
        
    Returns:
        tuple: A tuple containing the image data (X) and corresponding labels.
    """

    folder_path = "data/11t51center"

    labels_filepath = folder_path + "/Segmented"
    X_1_filepath = folder_path + "/Slicefront_corrected/Detector1"
    X_2_filepath = folder_path + "/Slicefront_corrected/Detector2"

    labels_filenames = os.listdir(labels_filepath)
    X_1_filenames = os.listdir(X_1_filepath)
    X_2_filenames = os.listdir(X_2_filepath)

    if verbose:
        print("\nLoading data...")

    labels = np.array([plt.imread(labels_filepath + "/" + filename) for filename in labels_filenames])
    X_1 = np.array([plt.imread(X_1_filepath + "/" + filename) for filename in X_1_filenames])
    X_2 = np.array([plt.imread(X_2_filepath + "/" + filename) for filename in X_2_filenames]) 

    if verbose:
        print("\nData loaded.")
    
    if processing:
        if verbose:
            print("\nProcessing data...")
        X_1 = X_1[1-1:646,357-1:900, 4-1:900]
        X_2 = X_2[1-1:646,357-1:900, 4-1:900]

        frame_to_delete = 542
        X_1 = np.delete(X_1, frame_to_delete, axis=0)
        X_2 = np.delete(X_2, frame_to_delete, axis=0)

        if verbose:
            print("\nData processed.\n")
    
    X = np.stack([X_1,X_2], axis=0)

    # One hot encode the labels


    return X, labels

def data_load_tensors(verbose:bool = True, processing:bool = True, one_hot:bool = False):
    """
    Load data into tensors and perform data processing.

    Args:
        verbose (bool): Whether to display verbose output.
        processing (bool): Whether to perform data processing.

    Returns:
        torch.Tensor: The input data in tensor format.
        torch.Tensor: The corresponding labels in tensor format.
    """
    X, y = data_load_numpy(verbose, processing)

    X, y = torch.tensor(X), torch.tensor(y)

    # Permute the dimensions of X to have the channel dimension as the second one
    X = X.permute(1, 0, 2, 3)

    # # Add singleton dimension
    # X = X.unsqueeze(1)
    # y = y.unsqueeze(0)

    class_mapping = {0: 0, 128: 1, 255: 2}
    mapped_labels = torch.zeros_like(y, dtype=torch.long)

    for original_class, mapped_class in class_mapping.items():
        mapped_labels[y == original_class] = mapped_class

    y = mapped_labels

    if one_hot:
        y = torch.nn.functional.one_hot(y, num_classes=3)
        y = y.permute(0, 3, 1, 2)
    # Permute the dimensions of y to have the channel dimension as the second one
    

    # # Convert X to float
    X = X.float()
    return X, y 

def get_dataloaders(batch_size:int=15, train_size:float = 0.8, seed:int = 42, verbose:bool = True):

    X, y = data_load_tensors(verbose)
    train_size = int(train_size * len(X))    
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]


    # dataset = TensorDataset(X, y)
    train_dataset = InMemoryDataset(X_train, y_train)
    test_dataset = InMemoryDataset(X_test, y_test)

    # train_size = int(0.8 * len(dataset))
    # test_size = len(dataset) - train_size
    # # generator = torch.Generator().manual_seed(seed)
    # # train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=True)

    return train_loader, test_loader

if __name__ == "__main__":
    # (X_1, X_2), y = data_load_tensors()

    X, y = data_load_tensors()

    # Create a dataset
    dataset = InMemoryDataset(X, y)

    # Get the first sample
    data, target = dataset[0]

    # Print the shapes
    print(data.shape)  # Should print: torch.Size([2, 544, 897])
    print(target.shape)  # Should print: torch.Size([544, 897])

    print("shape of X", X.shape)
    print("shape of y", y.shape)
    # train_loader, test_loader = get_dataloaders()

    print("done")