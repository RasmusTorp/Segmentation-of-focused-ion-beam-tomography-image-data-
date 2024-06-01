import matplotlib.pyplot as plt
import torch
import numpy as np
from evaluation.evaluation import calculate_iou, calculate_pixel_accuracy


def plot_all(model, testSet, slice, save_as):
    
    model.eval()
    
    X, y = testSet[slice]
    
    X = X.unsqueeze(0)
    
    pred = model.to_segmentation(X)
    y = y.unsqueeze(0)
    mean_iou = calculate_iou(pred, y)
    pixel_accuracy = calculate_pixel_accuracy(pred, y)
    
    pred = pred.squeeze(0).numpy()
    label = y.squeeze(0).numpy()
    X = X.squeeze(0).numpy()
    x1, x2 = X[0], X[1]
    
    fig, axs = plt.subplots(3, 2, figsize=(10, 15))  # Create a figure and a 3x2 subplot grid

    # Add space between subplots
    plt.subplots_adjust(wspace=0.2, hspace=-0.5) 
    
    
    plt.suptitle(f'Mean IoU: {mean_iou}, Pixel Accuracy: {pixel_accuracy}', fontsize=16)

    # Plot the Mask
    axs[0, 0].imshow(label)
    axs[0, 0].set_title('Mask')

    # Plot the prediction
    axs[0, 1].imshow(pred)
    axs[0, 1].set_title('Prediction')

    # Plot the detector 1
    axs[1, 0].imshow(x1)
    axs[1, 0].set_title('Detector 1')

    # Plot the detector 2
    axs[1, 1].imshow(x2)
    axs[1, 1].set_title('Detector 2')

    # Plot where prediction and label are not equal
    axs[2, 0].imshow(pred != label)
    axs[2, 0].set_title('Pred != Label')

    # Hide unused subplot
    axs[2, 1].axis('off')

    plt.savefig(f"plots/{save_as}_{slice}.png")
    plt.clf()