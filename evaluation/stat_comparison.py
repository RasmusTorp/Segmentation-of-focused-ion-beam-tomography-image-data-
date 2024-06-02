import torch
from scipy import stats
import numpy as np
from evaluation.evaluation import evaluate_model

def compare_models(model1, model2, test_loader, device):
    model1.to(device)
    model2.to(device)
    model1.eval()
    model2.eval()
    IOU1 = []
    IOU2 = []
    pixel_accuracy1 = []
    pixel_accuracy2 = []
    
    for i, (X, y) in enumerate(test_loader):
        with torch.no_grad():
            X, y = X.to(device), y.to(device)
            y_pred1 = model1.to_segmentation(X)
            y_pred2 = model2.to_segmentation(X)
            
            pix1, iou1 = evaluate_model(y_pred1, y, print_values=False)
            pix2, iou2 = evaluate_model(y_pred2, y, print_values=False)
            pix1, iou1 = pix1.item(), iou1.item()
            pix2, iou2 = pix2.item(), iou2.item()
            
            pixel_accuracy1.append(pix1)
            pixel_accuracy2.append(pix2)
            IOU1.append(iou1)
            IOU2.append(iou2)
            
    pixel_accuracy1 = np.array(pixel_accuracy1)
    pixel_accuracy2 = np.array(pixel_accuracy2)
    IOU1 = np.array(IOU1)
    IOU2 = np.array(IOU2)
    
    print(f"Pixel accuracy model 1: {pixel_accuracy1.mean()}, Pixel accuracy model 2: {pixel_accuracy2.mean()}")
    print(f"IOU model 1: {IOU1.mean()}, IOU model 2: {IOU2.mean()}")
    print(f"Pixel accuracy model 1 std: {pixel_accuracy1.std()}, Pixel accuracy model 2 std: {pixel_accuracy2.std()}")
    print(f"IOU model 1 std: {IOU1.std()}, IOU model 2 std: {IOU2.std()}")
    
    t_stat_pixel, p_value_pixel = stats.ttest_rel(pixel_accuracy1, pixel_accuracy2)
    t_stat_iou, p_value_iou = stats.ttest_rel(IOU1, IOU2)
    
    print(f"T-statistic_pixel: {t_stat_pixel}, P-value_pixel: {p_value_pixel}")
    print(f"T-statistic_iou: {t_stat_iou}, P-value_iou: {p_value_iou}")