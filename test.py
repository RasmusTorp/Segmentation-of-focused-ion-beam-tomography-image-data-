import subprocess
from omegaconf import OmegaConf
import hydra
import wandb
import os
from dataLoad import get_dataloaders
from UNET_2D import UNet2D
import torch
from plotting_functions.plot_all import plot_all
from plotting_functions.plot_trough_network import plot_trough_network
from evaluation import calculate_iou, calculate_pixel_accuracy

@hydra.main(config_name="config_test.yaml", config_path="./", version_base="1.3")
def main(config):
    if config.compute.hpc:
        folder_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir, "data", "data","11t51center"))
    
    else: 
        folder_path = "data/11t51center"
        
        
    train_loader, test_loader = get_dataloaders(batch_size=config.hyper.batch_size, train_size=config.data.train_size, 
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
    
    model.load_model(f"saved_models/{config.miscellaneous.save_as}", map_location=torch.device('cpu'))
    
    if config.testing.test_on == "test":
        dataset = test_loader.dataset    
        plot_all(model, dataset, config.testing.slice, "test")
    
    elif config.testing.test_on == "train":
        dataset = train_loader.dataset
        plot_all(model, dataset, config.testing.slice, "train")
        
    
    if config.testing.plot_trough_network:
        x, y  = dataset[config.testing.slice]
        x = x.unsqueeze(0)
        model.eval()
        model.plot_trough_network(x, save_as="trough_network")
    
    
    if config.testing.evaluate:
        model.evaluate(test_loader)
        
    print("done")
    
    
    
if __name__ == "__main__":
    main()