import matplotlib.pyplot as plt


fontSize = 20
titleSize = 25
point_offset = (7,-12)
legend_box_size = 20

# Data for NoFlip
# percentage = [10, 20, 30, 40, 50, 60, 70]
# iou_noflip = [0.87, 0.90, 0.92, 0.92, 0.92, 0.92, 0.90]
# accuracy_noflip = [0.93, 0.95, 0.96, 0.96, 0.96, 0.96, 0.95]
# val_loss_noflip = [0.23, 0.14, 0.11, 0.10, 0.10, 0.10, 0.13]


# Data for NoFlip
percentage = [10, 20, 30, 40, 50, 60, 70]
iou_noflip = [0.86, 0.91, 0.93, 0.92, 0.93, 0.93, 0.92]
accuracy_noflip = [0.93, 0.95, 0.96, 0.96, 0.96, 0.96, 0.96]


# # Data for withFlip
# iou_withflip = [0.83, 0.88, 0.91, 0.90, 0.92, 0.91, 0.91]
# accuracy_withflip = [0.91, 0.94, 0.96, 0.95, 0.96, 0.96, 0.95]
# val_loss_withflip = [0.29, 0.17, 0.13, 0.13, 0.10, 0.11, 0.14]

# Plotting
plt.figure(figsize=(12, 8))

# IOU data
plt.plot(percentage, iou_noflip, marker='o', label='IOU: Model 1', linestyle='-', color='blue')
for i, txt in enumerate(iou_noflip):
    plt.annotate(txt, (percentage[i], iou_noflip[i]),xytext=point_offset, textcoords='offset points')


# plt.plot(percentage, iou_withflip, marker='o', label='IOU: Model 2', linestyle='--', color='blue')

# Accuracy data
plt.plot(percentage, accuracy_noflip, marker='s', label='Accuracy: Model 1', linestyle='-', color='green')
for i, txt in enumerate(accuracy_noflip):
    plt.annotate(txt, (percentage[i], accuracy_noflip[i]),xytext=point_offset, textcoords='offset points')

# plt.plot(percentage, accuracy_withflip, marker='s', label='Accuracy: Model 2', linestyle='--', color='green')

# Val Loss data
# plt.plot(percentage, val_loss_noflip, marker='^', label='Val Loss: Model 1', linestyle='-', color='red')
# plt.plot(percentage, val_loss_withflip, marker='^', label='Val Loss: Model 2', linestyle='--', color='red')

plt.xlabel('Percentage of data used for training', fontsize = fontSize)
plt.ylabel('Value', fontsize = fontSize)
plt.ylim((0, 1))
plt.title('IOU and Pixel Accuracy Over Training Data Percentage', fontsize = titleSize)
plt.legend(loc = "lower right", prop={'size':legend_box_size})
plt.grid(True)
plt.xticks(percentage)
# plt.show()
plt.savefig('results/performance_over_train_size_10to70.pdf', format='pdf')
