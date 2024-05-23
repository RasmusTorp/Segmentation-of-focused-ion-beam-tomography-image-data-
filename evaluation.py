
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

# def calculate_iou(pred, target, num_classes = 3):
#     epsilon = 1e-8
#     ious = []
#     for cls in range(num_classes):  # Loop over each class
#         pred_inds = pred == cls
#         target_inds = target == cls
#         intersection = (pred_inds[target_inds]).sum()  # Intersection is where both are the current class
#         union = pred_inds.sum() + target_inds.sum() - intersection  # Union is either the current class, minus intersection
#         iou = (intersection + epsilon) / (union + epsilon)  # Add small epsilon to avoid division by zero
#         ious.append(iou)

#     return torch.stack(ious).mean()  # Return the average IoU over all classes


def calculate_iou(pred, target, num_classes=3):
    epsilon = 1e-6
    ious = []

    for cls in range(num_classes):  # Loop over each class
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum().float()  # Intersection is where both are the current class
        union = pred_inds.sum().float() + target_inds.sum().float() - intersection  # Union is either the current class, minus intersection

        if union == 0:
            iou = torch.tensor(1.0 if intersection == 0 else 0.0)  # If there is no union and no intersection, IoU is 1
        else:
            iou = intersection / union  # Compute IoU

        ious.append(iou)

    return torch.stack(ious).mean()  # Return the average IoU over all classes

# SMOOTH = 1e-6

# from: https://www.kaggle.com/code/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy
# def calculate_iou(outputs: torch.Tensor, labels: torch.Tensor):
#     # You can comment out this line if you are passing tensors of equal shape
#     # But if you are passing output from UNet or something it will most probably
#     # be with the BATCH x 1 x H x W shape
#     outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    
#     intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
#     union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
    
#     iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
#     thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
#     return thresholded.mean()  # Or thresholded.mean() if you are interested in average across the batch
    


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
