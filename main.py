import subprocess
from omegaconf import OmegaConf
import hydra
import wandb
import os
from dataLoad import get_dataloaders
from UNET_2D import UNet2D
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
    
    # Chooses 
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
                                                p_flip_horizontal=config.data.p_flip_horizontal,
                                                add_gaussian_noise=config.data.add_gaussian_noise,
                                                add_jitter=config.data.add_jitter,)

    n_channels = 2 if config.data.detector == "both" else 1
        
    model = UNet2D(n_neurons=config.hyper.n_neurons,
                    n_channels=n_channels, 
                    n_classes=config.constants.n_classes,
                    n_depth=config.hyper.n_depth,
                    with_skip_connections=config.hyper.with_skip_connections)
    
    model.train_model(train_loader = train_loader, val_loader=val_loader, epochs=config.hyper.epochs, 
                    lr=config.hyper.lr, patience=config.hyper.patience, track = config.wandb.track, save_as=config.miscellaneous.save_as)
    
    model.evaluate(test_loader)
    print("done")
    
if __name__ == "__main__":
    main()