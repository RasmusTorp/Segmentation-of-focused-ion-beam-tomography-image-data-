import torch
import numpy as np
import matplotlib.pyplot as plt
import os

folder_path = "data/Ming_cell"

def preprocess_ming(X, y):
    # Define the conditions for the mapping
    conditions = [np.isclose(y, 0., atol=1e-7), 
                np.isclose(y, 0.5058824, atol=1e-7), 
                np.isclose(y, 1., atol=1e-7)]

    # Define the corresponding values for the mapping
    values = [0, 1, 2]

    # Apply the mapping
    y = np.select(conditions, values)
    
    return torch.tensor(X), torch.tensor(y)
    

def load_ming(folder_path):
    labels_filepath = folder_path + "/All_phases_label_segmentation2G"

    # X1 and X2 are reversed in this dataset
    X_1_filepath = folder_path + "/Ming Raw aligned images/Slicefront corrected/Detector2_cropped"
    X_2_filepath = folder_path + "/Ming Raw aligned images/Slicefront corrected/Detector1_cropped"

    labels_filenames = sorted(os.listdir(labels_filepath))
    X_1_filenames = sorted(os.listdir(X_1_filepath))
    X_2_filenames = sorted(os.listdir(X_2_filepath))

    y = np.array([plt.imread(labels_filepath + "/" + filename) for filename in labels_filenames])
    X_1 = np.array([plt.imread(X_1_filepath + "/" + filename) for filename in X_1_filenames])
    X_2 = np.array([plt.imread(X_2_filepath + "/" + filename) for filename in X_2_filenames]) 
    
    X = np.stack([X_1,X_2], axis=0)
    
    return preprocess_ming(X, y)
    

if __name__ == "__main__":
    X, y = load_ming(folder_path)
    print(X.shape, y.shape)

print("done")