from dataLoad import data_load_tensors
import numpy as np
import torch
import torch.nn.functional as F

def calculate_pixel_accuracy(pred, target):
    # pred and target are of shape [batch_size, height, width]
    correct = (pred == target).sum()  # Count number of correct predictions
    total = target.numel()  # Get total number of pixels
    return correct.float() / total  # Return accuracy


# # Mean IoU (Intersection over Union)
# def calculate_iou(pred, target):
#     epsilon = 1e-8
#     # pred and target are of shape [batch_size, height, width]
#     intersection = ((pred == 1) & (target == 1)).sum(dim=(1,2))  # Sum over height and width where both are 1
#     union = ((pred == 1) | (target == 1)).sum(dim=(1,2))  # Sum over height and width where either is 1
#     iou = (intersection + epsilon) / (union + epsilon)  # Add small epsilon to avoid division by zero
#     return iou.mean()  # Return the average IoU over the batch

def calculate_iou(pred, target, num_classes = 3):
    epsilon = 1e-8
    ious = []
    for cls in range(num_classes):  # Loop over each class
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).sum()  # Intersection is where both are the current class
        union = pred_inds.sum() + target_inds.sum() - intersection  # Union is either the current class, minus intersection
        iou = (intersection + epsilon) / (union + epsilon)  # Add small epsilon to avoid division by zero
        ious.append(iou)

    return torch.stack(ious).mean()  # Return the average IoU over all classes

#TODO: Have function for surface area

# def phase_fraction(pred, target):
#     # pred and target are of shape [batch_size, height, width]
#     phase_0 = (pred == 0).sum()  # Count number of phase 0 pixels
#     phase_1 = (pred == 1).sum()  # Count number of phase 1 pixels
#     phase_2 = (pred == 2).sum()  # Count number of phase 2 pixels
#     total = target.numel()  # Get total number of pixels
#     return phase_0.float()/total, phase_1.float()/total, phase_2.float()/total  # Return phase fractions

def evaluate_model(pred, target, print_values = True, return_values = False):
    pixel_accuracy = calculate_pixel_accuracy(pred, target)
    mean_iou = calculate_iou(pred, target)

    if print_values:
        print(f'Pixel accuracy: {pixel_accuracy.item()}, Mean IoU: {mean_iou.item()}')

    return pixel_accuracy, mean_iou


if __name__ == "__main__":
    X, y = data_load_tensors()
    y_1, y_2 = y[0:15,:,:], y[15:30,:,:]
    
    print(mIOU(y_1, y_2))
    evaluate_model(y_1, y_2)