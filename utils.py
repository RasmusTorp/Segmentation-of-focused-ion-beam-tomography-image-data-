import matplotlib.pyplot as plt
import torch

def plot_slice(img, i, channel, show = True, save_as = False):
    plt.imshow(img[i, channel, :, :])

    if show:
        plt.show()

    if save_as:
        plt.savefig("plots/" + save_as)

def plot_label(img, slice_number):
    plt.imshow(img[slice_number])

def plot_slider(channel):
    pass

def get_device():
    if torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")
    
    if torch.backends.mps.is_available():
        print("Using MPS")
        return torch.device("mps")
    
    print("Using CPU")
    return torch.device("cpu")