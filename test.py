import subprocess
from omegaconf import OmegaConf
import hydra
import wandb
import os
from dataLoad import get_dataloaders, data_load_tensors
from UNET_2D import UNet2D
import torch
from plotting_functions.plot_all import plot_all
from evaluation import calculate_iou, calculate_pixel_accuracy
from segmentSlice import segment_slice

@hydra.main(config_name="config_test.yaml", config_path="./", version_base="1.3")
def main(config):
    if config.compute.hpc:
        folder_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir, "data", "data","11t51center"))
    
    else: 
        folder_path = "data/11t51center"
    
    
        
    train_loader, test_loader, val_loader = get_dataloaders(batch_size=config.hyper.batch_size, train_size=config.data.train_size,
                                                test_size=config.data.test_size, 
                                                seed=config.constants.seed, sampling_height=config.data.sampling_height, 
                                                sampling_width=config.data.sampling_width, 
                                                in_memory=config.data.in_memory, 
                                                static_test=config.data.static_test, folder_path=folder_path,
                                                random_train_test_split=config.data.random_train_test_split,
                                                detector=config.data.detector, normalize=config.data.normalize,
                                                p_flip_horizontal=config.data.p_flip_horizontal)

    n_channels = 2 if config.data.detector == "both" else 1
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = UNet2D(n_neurons=config.hyper.n_neurons,
                    n_channels=n_channels, 
                    n_classes=config.constants.n_classes,
                    n_depth=config.hyper.n_depth,
                    with_skip_connections=config.hyper.with_skip_connections)
    
    model.load_model(f"saved_models/{config.testing.model}", map_location=torch.device('cpu'))
    
    # X, y = data_load_tensors(folder_path=folder_path)
    
    # y_i = y[config.testing.slice]
    # segmentation = segment_slice(model, X[config.testing.slice], sampling_height=config.data.sampling_height, sampling_width=config.data.sampling_width, stride = 100)
    
    if config.testing.test_on == "test" and config.testing.plot:
        dataset = test_loader.dataset    
        plot_all(model, dataset, config.testing.slice, f"withflipping_test")
    
    elif config.testing.test_on == "train" and config.testing.plot:
        dataset = train_loader.dataset
        plot_all(model, dataset, config.testing.slice, f"withflipping_train")
        
    
    if config.testing.plot_trough_network:
        dataset = test_loader.dataset  
        x, y  = dataset[config.testing.slice]
        x = x.unsqueeze(0)
        model.eval()
        model.plot_trough_network(x, save_as=f"trough_network_withflipping_{config.testing.slice}")
    
    
    if config.testing.evaluate:
        model.evaluate(test_loader)
        
    print("done")
    
    
    
if __name__ == "__main__":
    main()