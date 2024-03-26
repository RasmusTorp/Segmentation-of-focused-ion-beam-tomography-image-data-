from dataLoad import data_load_numpy
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import Normalize

folder_path = "data/11t51center"

labels_filepath = folder_path + "/Segmented"
X_1_filepath = folder_path + "/Slicefront_corrected/Detector1"
X_2_filepath = folder_path + "/Slicefront_corrected/Detector2"

labels_filenames = sorted(os.listdir(labels_filepath))
X_1_filenames = sorted(os.listdir(X_1_filepath))
X_2_filenames = sorted(os.listdir(X_2_filepath))


labels = np.array([plt.imread(labels_filepath + "/" + filename) for filename in labels_filenames])
X_1 = np.array([plt.imread(X_1_filepath + "/" + filename) for filename in X_1_filenames])
X_2 = np.array([plt.imread(X_2_filepath + "/" + filename) for filename in X_2_filenames]) 

X_1 = X_1[:,357-1:900, 4-1:900]
X_2 = X_2[:,357-1:900, 4-1:900]

X_1 = np.delete(X_1, 541, axis=0)
X_2 = np.delete(X_2, 541, axis=0)

X_1 = X_1 / 255
X_2 = X_2 / 255

X_1 = torch.tensor(X_1)
X_2 = torch.tensor(X_2)

X_1_mean = X_1.mean()
X_2_mean = X_2.mean()

X_1_std = X_1.std()
X_2_std = X_2.std()

mean = torch.tensor([X_1_mean, X_2_mean])
std = torch.tensor([X_1_std, X_2_std])

normalize = Normalize(mean, std)


#! TODO does not work yet
X = torch.stack([X_1,X_2], axis=0)

X = normalize(X)

    
X = torch.permute(X, (1, 0, 2, 3))


# # For detector 2
# for i, image in enumerate(X_2):
#     if i == 541:
#         continue
    
#     filename = X_2_filenames[i]
#     new_filename = os.path.splitext(filename)[0] + '.png'  # Change the file extension to .png
#     plt.imsave(f"data/11t51center/Slicefront_corrected/Detector2_cropped/{new_filename}", image, cmap='gray')

print("done")