from loadDataFunctions.loadMing import load_ming
from loadDataFunctions.load11t51center import data_load_tensors
import matplotlib.pyplot as plt
import numpy as np
from dataLoad import get_normalizer
from scipy.stats import norm


import torch

folder_path_ming = "data/Ming_cell"

X1, y1 = data_load_tensors()
X2, y2 = load_ming(folder_path_ming)


normalizer1 = get_normalizer(X1)
normalizer2 = get_normalizer(X2)

X1 = normalizer1(X1).numpy().flatten()
X2 = normalizer2(X2).numpy().flatten() 

# Create histograms
plt.figure(figsize=(12, 6))

# Histogram for the first image
plt.subplot(1, 2, 1)
plt.hist(X1, bins=50, color='blue', alpha=0.7)
plt.title('Histogram of Dataset 1')
plt.xlabel('Intensity Values')
plt.ylabel('Frequency')


# Histogram for the second image
plt.subplot(1, 2, 2)
plt.hist(X2, bins=50, color='green', alpha=0.7)
plt.title('Histogram of Dataset 2')
plt.xlabel('Intensity Values')
plt.ylabel('Frequency')




print(f"Mean of Dataset 1: {np.mean(X1)}, Mean of Dataset 2: {np.mean(X2)}")
print(f"Std of Dataset 1: {np.std(X1)}, Std of Dataset 2: {np.std(X2)}")


plt.tight_layout()
plt.savefig('results/histograms.pdf', format='pdf')
# plt.show()