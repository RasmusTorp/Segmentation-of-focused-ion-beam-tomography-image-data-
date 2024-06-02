# import subprocess
# from omegaconf import OmegaConf
import hydra
# import wandb
import os

from dataLoad import get_dataloaders
from loadDataFunctions.load11t51center import data_load_tensors
from loadDataFunctions.loadMing import load_ming
from UNET_2D import UNet2D
import torch
from plotting_functions.plot_all import plot_all
from evaluation.stat_comparison import compare_models
# from evaluation import calculate_iou, calculate_pixel_accuracy
# from segmentSlice import segment_slice

@hydra.main(config_name="config_test.yaml", config_path="./", version_base="1.3")
def main(config):
    if config.compute.hpc:
        folder_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir, "data", "data", config.data.dataset))
    
    else: 
        folder_path = f"data/{config.data.dataset}"
    
    
    if config.data.dataset=="Ming_cell":
        X, y = load_ming(folder_path)
        
    elif config.data.dataset=="11t51center":
        X, y = data_load_tensors(folder_path=folder_path)
    
        
    train_loader, test_loader, val_loader = get_dataloaders(X, y, batch_size=config.hyper.batch_size, train_size=config.data.train_size,
                                                test_size=config.data.test_size, 
                                                seed=config.constants.seed, sampling_height=config.data.sampling_height, 
                                                sampling_width=config.data.sampling_width, 
                                                static_test=config.data.static_test,
                                                random_train_test_split=config.data.random_train_test_split,
                                                detector=config.data.detector,
                                                p_flip_horizontal=config.data.p_flip_horizontal)

    n_channels = 2 if config.data.detector == "both" else 1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
        plot_all(model, dataset, config.testing.slice, f"{config.testing.model}_{config.data.dataset}_test")
    
    elif config.testing.test_on == "train" and config.testing.plot:
        dataset = train_loader.dataset
        plot_all(model, dataset, config.testing.slice, f"{config.testing.model}_{config.data.dataset}_train")
        
    
    if config.testing.plot_trough_network:
        dataset = test_loader.dataset  
        x, y  = dataset[config.testing.slice]
        x = x.unsqueeze(0)
        model.eval()
        model.plot_trough_network(x, save_as=f"trough_network_{config.testing.model}_{config.testing.slice}")
    
    if config.testing.compare_with:
        model2 = UNet2D(n_neurons=config.hyper.n_neurons,
                    n_channels=n_channels, 
                    n_classes=config.constants.n_classes,
                    n_depth=config.testing.compare_with_depth,
                    with_skip_connections=config.hyper.with_skip_connections)
        
        model2.load_model(f"saved_models/{config.testing.compare_with}", map_location=device)
        print(f"Comparing models {config.testing.model} and {config.testing.compare_with}:")
        compare_models(model, model2, test_loader, device)
        
        print(f"Model {config.testing.model} performance:")
        model.evaluate(test_loader)
        
        print(f"Model {config.testing.compare_with} performance:")
        model2.evaluate(test_loader)
    
    
    if config.testing.evaluate and not config.testing.compare_with:
        model.evaluate(test_loader)
        
    print("done")
    
    
    
if __name__ == "__main__":
    main()