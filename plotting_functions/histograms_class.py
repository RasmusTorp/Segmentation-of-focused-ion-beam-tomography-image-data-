from loadDataFunctions.loadMing import load_ming
from loadDataFunctions.load11t51center import data_load_tensors
import matplotlib.pyplot as plt
import numpy as np
from dataLoad import get_normalizer
from scipy.stats import norm


import torch

folder_path_ming = "data/Ming_cell"
title_size = 20
label_size = 18

X1, y1 = data_load_tensors()

X2, y2 = load_ming(folder_path_ming)


normalizer1 = get_normalizer(X1)
normalizer2 = get_normalizer(X2)

X1 = normalizer1(X1).numpy()
X2 = normalizer2(X2).numpy()

X1 = X1[:,0,:,:].flatten()
X2 = X2[:,0,:,:].flatten() 

y1 = y1.numpy().flatten()
y2 = y2.numpy().flatten()

class1 = [X1[y1==i] for i in range(3)]
class2 = [X2[y2==i] for i in range(3)]

# Create histograms
fig, axs = plt.subplots(2, 3, figsize=(18, 12))

class_names = ["Pore", "YSZ", "Nickel"]

# Histogram for the first dataset
for i in range(3):
    plt.subplot(2, 3, i+1)
    plt.hist(class1[i], bins=50, color='blue', alpha=0.7)
    plt.title(f'Histogram of Dataset 1, {class_names[i]}', fontsize=title_size)
    plt.xlabel('Intensity Values', fontsize=label_size)
    plt.ylabel('Frequency', fontsize=label_size)

# Histogram for the second dataset
for i in range(3):
    plt.subplot(2, 3, i+4)
    plt.hist(class2[i], bins=50, color='green', alpha=0.7)
    plt.title(f'Histogram of Dataset 2, {class_names[i]}', fontsize=title_size)
    plt.xlabel('Intensity Values', fontsize=label_size)
    plt.ylabel('Frequency', fontsize=label_size)

fig.subplots_adjust(hspace=0.35, wspace=0.2)

plt.tight_layout()
plt.savefig('results/histograms_classes.pdf', format='pdf')
# plt.show()