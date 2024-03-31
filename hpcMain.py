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
    
    
    folder_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir, "data", "data","11t51center"))
    
    train_loader, test_loader = get_dataloaders(batch_size=config.hyper.batch_size, train_size=config.hyper.train_size, 
                                                square_size=config.hyper.square_size, in_memory=config.hyper.in_memory, 
                                                static_test=config.hyper.static_test, folder_path=folder_path)

    model = UNet2D(n_neurons=config.hyper.n_neurons,
                    n_channels=config.constants.n_channels,
                    n_classes=config.constants.n_classes,
                    n_depth=config.hyper.n_depth,
                    with_skip_connections=config.hyper.with_skip_connections)
    
    model.train_model(train_loader = train_loader, test_loader=test_loader, epochs=config.hyper.epochs, 
                    lr=config.hyper.lr, patience=config.hyper.patience)
    
    model.evaluate(test_loader)
    
    print("done")
    
    
if __name__ == "__main__":
    main()