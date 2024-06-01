import os
import matplotlib.pyplot as plt
import numpy as np
import torch


def data_load_numpy(verbose:bool = True, processing = False, folder_path = "data/11t51center"):
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

    return X, labels

def data_load_tensors(verbose:bool = True, processing:bool = True, one_hot:bool = False, map_lapels = True, folder_path = "data/11t51center"):
    """
    Load data into tensors and perform data processing.

    Args:
        verbose (bool): Whether to display verbose output.
        processing (bool): Whether to perform data processing.

    Returns:
        torch.Tensor: The input data in tensor format.
        torch.Tensor: The corresponding labels in tensor format.
    """
    X, y = data_load_numpy(verbose, processing, folder_path=folder_path)

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


if __name__ == "__main__":
    X, y = data_load_tensors()
    
    
    # Count 0, 1 and 2 in y and divide by total number of pixels
    print(f"Class 0: {torch.sum(y==0).item() / y.numel()}")
    print(f"Class 1: {torch.sum(y==1).item() / y.numel()}")
    print(f"Class 2: {torch.sum(y==2).item() / y.numel()}")
    
    