import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

model_name_1 = "noflip50"
model_name_2 = "noflip5"

# Set global parameters
plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=15)    # fontsize of the x and y labels
plt.rc('legend', fontsize=20)    # fontsize of the legend


acc1 = pd.read_csv(f"results/{model_name_1}_acc.csv").iloc[:,1]
iou1 = pd.read_csv(f"results/{model_name_1}_IOU.csv").iloc[:,1]
train_loss1 = pd.read_csv(f"results/{model_name_1}_train_loss.csv").iloc[:,1]
val_loss1 = pd.read_csv(f"results/{model_name_1}_val_loss.csv").iloc[:,1]

acc2 = pd.read_csv(f"results/{model_name_2}_acc.csv").iloc[:,1]
iou2 = pd.read_csv(f"results/{model_name_2}_IOU.csv").iloc[:,1]
train_loss2 = pd.read_csv(f"results/{model_name_2}_train_loss.csv").iloc[:,1]
val_loss2 = pd.read_csv(f"results/{model_name_2}_val_loss.csv").iloc[:,1]


# Plot acc1, acc2 and iou1, iou2
plt.figure(figsize=(10, 5))
plt.plot(acc1, label='Accuracy: Model 1', linestyle='-', color='blue')
plt.plot(iou1, label='IOU: Model 1', linestyle='-', color='green')
plt.plot(acc2, label='Accuracy: Model 2', linestyle='--', color='blue')
plt.plot(iou2, label='IOU: Model 2', linestyle='--', color='green')
plt.title('Accuracy and IOU Through Training Epochs')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.savefig(f"results/{model_name_1}_{model_name_2}_acc_iou.pdf", format="pdf")

# Plot train_loss1, val_loss1 and train_loss2, val_loss2
plt.figure(figsize=(10, 5))
plt.plot(train_loss1, label='Train Loss: Model 1', linestyle='-', color='orange')
plt.plot(val_loss1, label='Validation Loss: Model 1', linestyle='-', color='blue')
plt.plot(train_loss2, label='Train Loss: Model 2', linestyle='--', color='orange')
plt.plot(val_loss2, label='Validation Loss: Model 2', linestyle='--', color='blue')
plt.title('Train Loss and Validation Loss Through Training Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f"results/{model_name_1}_{model_name_2}_loss.pdf", format="pdf")
print("done")