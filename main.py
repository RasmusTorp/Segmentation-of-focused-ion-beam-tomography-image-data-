import subprocess
from omegaconf import OmegaConf
import hydra
import wandb
import os
from dataLoad import get_dataloaders
from UNET_2D import UNet2D
from loadDataFunctions.loadMing import load_ming
from loadDataFunctions.load11t51center import data_load_tensors
import torch

with open("secret.txt", "r") as f:
    os.environ['WANDB_API_KEY'] = f.read().strip()

@hydra.main(config_name="config.yaml", config_path="./", version_base="1.3")
def main(config):
    if config.wandb.track:
        print(f"configuration: \n {OmegaConf.to_yaml(config)}")
        # Initiate wandb logger
        try:
            # project is the name of the project in wandb, entity is the username
            # You can also add tags, group etc.
            run = wandb.init(project=config.wandb.project, 
                    config=OmegaConf.to_container(config), 
                    entity=config.wandb.entity)
            print(f"wandb initiated with run id: {run.id} and run name: {run.name}")
        except Exception as e:
            print(f"\nCould not initiate wandb logger\nError: {e}")
    
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
                                                in_memory=config.data.in_memory, 
                                                static_test=config.data.static_test, folder_path=folder_path,
                                                random_train_test_split=config.data.random_train_test_split,
                                                detector=config.data.detector, normalize=config.data.normalize,
                                                p_flip_horizontal=config.data.p_flip_horizontal)

    del X, y # Free up memory

    n_channels = 2 if config.data.detector == "both" else 1
    
    model = UNet2D(n_neurons=config.hyper.n_neurons,
                    n_channels=n_channels, 
                    n_classes=config.constants.n_classes,
                    n_depth=config.hyper.n_depth,
                    with_skip_connections=config.hyper.with_skip_connections)
    
    
    if config.hyper.hotstart_model:
        if config.compute.hpc:
            model.load_model(f"saved_models/{config.hyper.hotstart_model}")
        
        else:
            model.load_model(f"saved_models/{config.hyper.hotstart_model}", map_location=torch.device('cpu'))
    
    
    model.train_model(train_loader = train_loader, val_loader=val_loader, epochs=config.hyper.epochs, 
                    lr=config.hyper.lr, patience=config.hyper.patience, track = config.wandb.track, save_as=config.miscellaneous.save_as)
    
    pixel_accuracy, mean_iou = model.evaluate(test_loader)
    wandb.track({"pixel_accuracy_test": pixel_accuracy, "mean_iou_test": mean_iou})
    print("done")
    
if __name__ == "__main__":
    main()