import matplotlib.pyplot as plt


fontSize = 20
titleSize = 25

# Data for NoFlip
percentage_1to10 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
iou_noflip_1to10 = [0.63, 0.75, 0.82, 0.83, 0.85, 0.87, 0.84, 0.79, 0.82, 0.87]
accuracy_noflip_1to10 = [0.77, 0.86, 0.90, 0.91, 0.92, 0.93, 0.92, 0.89, 0.90, 0.93]
val_loss_noflip_1to10 = [0.74, 0.43, 0.31, 0.28, 0.24, 0.23, 0.27, 0.33, 0.31, 0.23]

percentage_10to70 = [10, 20, 30, 40, 50, 60, 70]
iou_noflip_10to70 = [0.87, 0.90, 0.92, 0.92, 0.92, 0.92, 0.90]
accuracy_noflip_10to70 = [0.93, 0.95, 0.96, 0.96, 0.96, 0.96, 0.95]
val_loss_noflip_10to70 = [0.23, 0.14, 0.11, 0.10, 0.10, 0.10, 0.13]

# Data for withFlip
iou_withflip_1to10 = [0.68, 0.76, 0.77, 0.83, 0.85, 0.85, 0.85, 0.82, 0.84, 0.83]
accuracy_withflip_1to10 = [0.81, 0.86, 0.87, 0.91, 0.92, 0.92, 0.92, 0.90, 0.92, 0.91]
val_loss_withflip_1to10 = [0.56, 0.40, 0.40, 0.31, 0.23, 0.25, 0.24, 0.28, 0.24, 0.29]

iou_withflip_10to70 = [0.83, 0.88, 0.91, 0.90, 0.92, 0.91, 0.91]
accuracy_withflip_10to70 = [0.91, 0.94, 0.96, 0.95, 0.96, 0.96, 0.95]
val_loss_withflip_10to70 = [0.29, 0.17, 0.13, 0.13, 0.10, 0.11, 0.14]

# Combine data
percentage = percentage_1to10 + percentage_10to70[1:]
iou_noflip = iou_noflip_1to10 + iou_noflip_10to70[1:]
accuracy_noflip = accuracy_noflip_1to10 + accuracy_noflip_10to70[1:]
val_loss_noflip = val_loss_noflip_1to10 + val_loss_noflip_10to70[1:]

iou_withflip = iou_withflip_1to10 + iou_withflip_10to70[1:]
accuracy_withflip = accuracy_withflip_1to10 + accuracy_withflip_10to70[1:]
val_loss_withflip = val_loss_withflip_1to10 + val_loss_withflip_10to70[1:]

# Plotting
plt.figure(figsize=(12, 8))

# IOU data
plt.plot(percentage, iou_noflip, marker='o', label='IOU NoFlip', linestyle='-', color='blue')
plt.plot(percentage, iou_withflip, marker='o', label='IOU withFlip', linestyle='--', color='blue')

# Accuracy data
plt.plot(percentage, accuracy_noflip, marker='s', label='Accuracy NoFlip', linestyle='-', color='green')
plt.plot(percentage, accuracy_withflip, marker='s', label='Accuracy withFlip', linestyle='--', color='green')

# Val Loss data
plt.plot(percentage, val_loss_noflip, marker='^', label='Val Loss NoFlip', linestyle='-', color='red')
plt.plot(percentage, val_loss_withflip, marker='^', label='Val Loss withFlip', linestyle='--', color='red')

plt.xlabel('Percentage of data used for training', fontsize = fontSize)
plt.ylabel('Value', fontsize = fontSize)
plt.ylim((0, 1))
# plt.title('IOU, Accuracy, and Val Loss Over Training Data Percentage', fontsize = titleSize)
plt.legend()
# plt.grid(True)
plt.xticks(percentage)

plt.savefig('results/performance_over_train_size_combined.pdf', format='pdf')

