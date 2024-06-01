import numpy as np
import matplotlib.pyplot as plt

def plot_sampling_heatmap(square_size, save_as = False):
    height, width = 544, 897
    image = np.zeros(shape=(height, width))

    # Calculate the probability for each position
    for i in range(height):
        for j in range(width):
            for h in range(height):
                if h <= i and i <= h + square_size and h + square_size <= height:
                    image[i, j] += 1

            for w in range(width):
                if w <= j and j <= w + square_size and w + square_size <= width:
                    image[i, j] += 1

    # Plot the heatmap
    plt.imshow(image, cmap='hot', interpolation='nearest')
    plt.colorbar()

    if save_as:
        plt.savefig("plots/" + save_as)

    plt.show()

plot_sampling_heatmap(256, "HeatMap256")