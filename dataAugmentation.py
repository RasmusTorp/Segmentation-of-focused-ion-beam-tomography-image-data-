import torchvision.transforms as v2
import torch
import numpy as np
import matplotlib.pyplot as plt
from loadDataFunctions.load11t51center import data_load_tensors
import torchvision.transforms.functional as F

class DataAugmentation:
    def __init__(self, gaussian_kernel_size, gaussian_sigma, brightness, contrast, p_flip_horizontal, sampling_width, sampling_height, random_sampling_train):
        if gaussian_kernel_size:
            self.gaussian_augmentation = v2.GaussianBlur(kernel_size=gaussian_kernel_size, sigma = (0.001, gaussian_sigma))
        else:
            self.gaussian_augmentation = None
    
        self.brightness = brightness
        self.contrast = contrast
        self.gaussian_sigma = gaussian_sigma
        self.gaussian_kernel_size = gaussian_kernel_size
        self.p_flip_horizontal = p_flip_horizontal
        self.sampling_width = sampling_width
        self.sampling_height = sampling_height
        self.random_sampling_train = random_sampling_train
        
    def get_random_square(self, X, y):
        h, w = X.shape[-2:]

        start_h = np.random.randint(0, h - self.sampling_height)
        start_w = np.random.randint(0, w - self.sampling_width)
        
        X = X[:, start_h:start_h+self.sampling_height, start_w:start_w+self.sampling_width]
        y = y[start_h:start_h+self.sampling_height, start_w:start_w+self.sampling_width]
        return X, y

    def flip_horizontal(self, X, y):
        x_channels = X.shape[0]
        X = torch.flip(X, [x_channels])  # Flip horizontally
        y = torch.flip(y, [1])  # Flip horizontally
        return X, y
    
    def brightness_augmentation(self, X):
        brightness_factor = 1 + torch.empty(1).uniform_(-self.brightness, self.brightness).item()
        # brightness_factor = 0.5
        X = X / 255
        augmented = F.adjust_brightness(X, brightness_factor=brightness_factor)
        return augmented * 255
        # return X * brightness_factor
    
    def contrast_augmentation(self, X):
        contrast_factor = 1 + torch.empty(1).uniform_(-self.contrast, self.contrast).item()
        # contrast_factor = 1.5
        X = X / 255
        augmented = F.adjust_contrast(X, contrast_factor=contrast_factor)
        return augmented * 255

    
    def __call__(self, X, y):
        
        # Random cropping
        if self.random_sampling_train:
            X, y = self.get_random_square(X, y)
        
        if self.p_flip_horizontal > 0:
            if np.random.rand() < self.p_flip_horizontal:
                X, y = self.flip_horizontal(X, y)
        
        if self.contrast:
            # Can only apply augmentations to one image at a time. 
            X = torch.stack([self.contrast_augmentation(x.unsqueeze(0)).squeeze(0) for x in X])
        
        if self.brightness:
            # Can only apply augmentations to one image at a time. 
            X = torch.stack([self.brightness_augmentation(x.unsqueeze(0)).squeeze(0) for x in X])
        
        if self.gaussian_augmentation:
            X = self.gaussian_augmentation(X)
        
        return X, y
    
    
    # Only used for a plot in the report or for debugging
    def plotAugmentations(self, X, y):
        fig, axs = plt.subplots(2, 3, figsize=(20, 10))  # Create a subplot with 1 row and 6 columns
        titleSize = 25
        subImageTitleSize = 18
        
        # Original Image
        axs[0, 0].imshow(X[0])  # Assuming X[0] is a 2D image
        axs[0, 0].set_title('Original Image',fontsize = subImageTitleSize)

        # Random cropping
        if self.random_sampling_train:
            X_crop, y_crop = self.get_random_square(X, y)
            axs[0, 1].imshow(X_crop[0], vmin=0, vmax=255)  # Plot X after random cropping
            axs[0, 1].set_title('Random Cropping',fontsize = subImageTitleSize)

        if self.p_flip_horizontal:
            X_flip, y_flip = self.flip_horizontal(X, y)
            axs[0, 2].imshow(X_flip[0], vmin=0, vmax=255)  # Plot X after horizontal flipping
            axs[0, 2].set_title('Horizontal Flipping',fontsize = subImageTitleSize)

        # Photometric transforms
        if self.contrast:
            # Can only apply augmentations to one image at a time. 
            # contrast_augmentation = v2.ColorJitter(contrast=self.contrast)
            X_contrast = torch.stack([self.contrast_augmentation(x.unsqueeze(0)).squeeze(0) for x in X])
            axs[1, 0].imshow(X_contrast[0].numpy(), vmin=0, vmax=255)  # Plot X after photometric transforms
            axs[1, 0].set_title('Contrast Augmentation',fontsize = subImageTitleSize)
            
        if self.brightness:
            # Can only apply augmentations to one image at a time. 
            # brightness_augmentation = v2.ColorJitter(brightness=self.brightness)
            X_bright = torch.stack([self.brightness_augmentation(x.unsqueeze(0)).squeeze(0) for x in X])
            axs[1, 1].imshow(X_bright[0].numpy(), vmin=0, vmax=255)
            axs[1, 1].set_title('Brightness Augmentation',fontsize = subImageTitleSize)

        # Gaussian augmentation
        if self.gaussian_augmentation:
            maxBlur = v2.GaussianBlur(kernel_size=self.gaussian_kernel_size, sigma = (self.gaussian_sigma -0.001, self.gaussian_sigma))
            X_gauss = maxBlur(X)
            axs[1, 2].imshow(X_gauss[0].numpy(), vmin=0, vmax=255)  # Plot X after Gaussian augmentation
            axs[1, 2].set_title('Gaussian Augmentation',fontsize = subImageTitleSize)

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        fig.suptitle("All Data Augmentations", fontsize = titleSize)
        plt.savefig('plots/all_transformations.png')  # Save the entire figure
        plt.show()
        

if __name__ == "__main__":
    from utils import get_normalizer
    X, y = data_load_tensors()
    dataAugmentations = DataAugmentation(gaussian_kernel_size=7, gaussian_sigma=0.05, brightness=0.5, contrast=0.5, p_flip_horizontal=0.5, sampling_height=272, sampling_width=448, random_sampling_train=True)
    normalizer = get_normalizer(X)
    
    X_i, y_i = X[0], y[0]

    X_i_aug, y_i_aug = dataAugmentations(X_i, y_i)
    
        
    X_i_aug = normalizer(X_i_aug)    
    
    print("done")