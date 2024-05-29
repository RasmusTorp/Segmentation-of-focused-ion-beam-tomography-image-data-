import matplotlib.pyplot as plt
import os


folder_path = "data/11t51center"

labels_filepath = folder_path + "/Segmented"
X_1_filepath = folder_path + "/Slicefront_corrected/Detector1"
X_2_filepath = folder_path + "/Slicefront_corrected/Detector2"

labels_filenames = sorted(os.listdir(labels_filepath))
X_1_filenames = sorted(os.listdir(X_1_filepath))
X_2_filenames = sorted(os.listdir(X_2_filepath))

plt.imshow(plt.imread(labels_filepath + "/" + labels_filenames[0]))
plt.show()