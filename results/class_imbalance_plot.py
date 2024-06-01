
import matplotlib.pyplot as plt

# 11t51center
Class_0 = 0.24410076960186994
Class_1 = 0.49554264201337894
Class_2 = 0.2603565883847511

# Set global parameters
plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=15)    # fontsize of the x and y labels
plt.rc('legend', fontsize=20)    # fontsize of the legend


# plot histogram
plt.figure(figsize=(10, 5))
plt.bar(['Class 0', 'Class 1', 'Class 2'], [Class_0, Class_1, Class_2])
plt.title('Class as a percentage of total data')
plt.xlabel('Class')
plt.ylabel('Percentage')
plt.savefig('results/class_imbalance_plot.pdf', format='pdf')
print("done")