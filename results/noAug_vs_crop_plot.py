import matplotlib.pyplot as plt

fontSize = 20
titleSize = 25
point_offset = (7,-15)
legend_box_size = 20
with_annotations = False

y_lim = (0.65, 1)

# Data for NoFlip
percentage = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
iou_noflip = [0.80, 0.76, 0.83, 0.82, 0.85, 0.86, 0.85, 0.81, 0.84, 0.86]
accuracy_noflip = [0.86, 0.86, 0.91, 0.90, 0.92, 0.92, 0.92, 0.89, 0.91, 0.93]

# val_loss_noflip = [0.74, 0.43, 0.31, 0.28, 0.24, 0.23, 0.27, 0.33, 0.31, 0.23]

# Data for crop
percentage_crop = [   1,    2,    3,    4,    5,    6,    7,    8,    9,   10]
iou_crop =        [0.76, 0.80, 0.85, 0.87, 0.88, 0.88, 0.90, 0.89, 0.91, 0.90]
accuracy_crop =   [0.86, 0.89, 0.92, 0.93, 0.94, 0.93, 0.95, 0.94, 0.95, 0.95]


# Plotting
plt.figure(figsize=(12, 8))

# IOU data
plt.plot(percentage, iou_noflip, marker='o', label='IOU: No Augmentation', linestyle='-', color='blue')
if with_annotations:
    for i, txt in enumerate(iou_noflip):
        plt.annotate(txt, (percentage_crop[i], iou_noflip[i]),xytext=point_offset, textcoords='offset points')

plt.plot(percentage_crop, iou_crop, marker='o', label='IOU: Random Crop', linestyle='--', color='blue')
if with_annotations:
    for i, txt in enumerate(iou_crop):
        plt.annotate(txt, (percentage[i], iou_crop[i]),xytext=point_offset, textcoords='offset points')

# Accuracy data
plt.plot(percentage, accuracy_noflip, marker='s', label='Accuracy: No Augmentation', linestyle='-', color='green')
if with_annotations:
    for i, txt in enumerate(accuracy_noflip):
        plt.annotate(txt, (percentage[i], accuracy_noflip[i]),xytext=point_offset, textcoords='offset points')
        
plt.plot(percentage_crop, accuracy_crop, marker='s', label='Accuracy: Random Crop', linestyle='--', color='green')
if with_annotations:
    for i, txt in enumerate(accuracy_crop):
        plt.annotate(txt, (percentage_crop[i], accuracy_crop[i]),xytext=point_offset, textcoords='offset points')
    

plt.xlabel('Percentage of data used for training', fontsize = fontSize)
plt.ylabel('Value', fontsize = fontSize)
plt.ylim(y_lim)
plt.title('Performance of Standard Model and Random Cropping ', fontsize = titleSize)
plt.legend(loc = "lower right", prop={'size':legend_box_size})
plt.grid(True)
plt.xticks(percentage)
plt.savefig('results/performance_over_train_size_1to10_crop_vs_noAug.pdf', format='pdf')

