
from dataLoad import data_load_tensors, get_dataloaders
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_device
from evaluation import evaluate_model
import numpy as np
import wandb

# Some parts partly inspired by https://github.com/ptrblck/pytorch_misc/blob/master/unet_demo.py and github copilot as well as original U-Net paper
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class DownStep(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding,
                stride):
        super(DownStep, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv = DoubleConv(in_channels, out_channels, kernel_size,
                                padding, stride)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x
    
class UpStep(nn.Module):
    def __init__(self, in_channels, in_channels_skip, out_channels,
                kernel_size, padding, stride):
        super(UpStep, self).__init__()

        self.conv_trans = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=2, padding=0, stride=2)
        
        self.conv_block = DoubleConv(
            in_channels=in_channels + in_channels_skip,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride)

    def forward(self, x, x_skip):
        x = self.conv_trans(x)
        x = torch.cat((x, x_skip), dim=1)
        x = self.conv_block(x)
        return x
    
class UNet2D(nn.Module):
    def __init__(self, n_neurons, n_channels, n_classes, n_depth = 3, kernel_size = 3, 
                padding = 1, stride = 1, with_skip_connections=True):
        super().__init__()
        
        self.with_skip_connections = with_skip_connections
        
        if with_skip_connections:
            self.encoder = nn.ModuleList()
            self.decoder = nn.ModuleList()
            
            first_conv = DoubleConv(n_channels, n_neurons, kernel_size=kernel_size, padding=padding, stride=stride)
            
            self.encoder.append(first_conv)
            for i in range(n_depth):
                self.encoder.append(DownStep(n_neurons * 2**i, n_neurons * 2**(i+1), kernel_size=kernel_size, padding=padding, stride=stride))
                
            for i in reversed(range(n_depth)):
                self.decoder.append(UpStep(n_neurons * 2**(i+1), n_neurons * 2**i, n_neurons * 2**i, kernel_size=kernel_size, padding=padding, stride=stride))
                
                
        else:
            self.encoder = nn.Sequential(
                DoubleConv(n_channels, n_neurons, kernel_size=kernel_size, padding=padding, stride=stride),
                nn.MaxPool2d(kernel_size=2),
                DoubleConv(n_neurons, n_neurons * 2, kernel_size=kernel_size, padding=padding, stride=stride),
                nn.MaxPool2d(kernel_size=2),
                DoubleConv(n_neurons * 2, n_neurons * 4, kernel_size=kernel_size, padding=padding, stride=stride),
                nn.MaxPool2d(kernel_size=2),
            )

            self.decoder = nn.Sequential(
                DoubleConv(n_neurons * 4, n_neurons * 2, kernel_size=kernel_size, padding=padding, stride=stride),
                nn.ConvTranspose2d(n_neurons * 2, n_neurons * 2, kernel_size=2, stride=2),
                DoubleConv(n_neurons * 2, n_neurons, kernel_size=kernel_size, padding=padding, stride=stride),
                nn.ConvTranspose2d(n_neurons, n_neurons, kernel_size=2, stride=2),
                DoubleConv(n_neurons, n_neurons, kernel_size=kernel_size, padding=padding, stride=stride),
                nn.ConvTranspose2d(n_neurons, n_neurons, kernel_size=2, stride=2),
                DoubleConv(n_neurons, n_classes, kernel_size=kernel_size, padding=padding, stride=stride),
            )
        
        self.n_classes = n_classes
        self.criterion = None

        self.optimizer = None
        self.device = get_device()

        self.final_conv = nn.Conv2d(in_channels=n_neurons, out_channels=self.n_classes, kernel_size=1)

    def forward(self, x):
        
        if not self.with_skip_connections:
            x = self.encoder(x)
            x = self.decoder(x)
            x = self.final_conv(x)

        else:
            encoder_outs = []
            
            for layer in self.encoder:
                x = layer(x)
                encoder_outs.append(x)
                
            for i, layer in enumerate(self.decoder):
                x = layer(x, encoder_outs[-i-2])
                
            x = self.final_conv(x)
            
        return x
    
    def to_segmentation(self, x):
        x = self.forward(x)
        return x.argmax(dim=1)

    def train_model(self, train_loader, test_loader = None, optimizer = "adam", lr = 0.001
                    ,criterion = "crossentropy", epochs = 10, verbose=2, patience=5, save_as="best_model.pth", track = False):
        
        self.to(self.device)
    
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        if criterion == "crossentropy":
            self.criterion = nn.CrossEntropyLoss()
        
        best_loss = np.inf
        no_improve_epochs = 0

        for epoch in range(epochs):
            total_loss = 0
            loss_calculated = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                loss_calculated += 1

                if verbose == 2:
                    print("loss: ", loss.item())


                loss.backward()
                self.optimizer.step()

            if verbose:
                print(f'Train Epoch: {epoch + 1}, Loss: {total_loss / loss_calculated}')
                
            
            if track:
                wandb.log({"train_loss": total_loss / loss_calculated})

            if test_loader:
                val_loss = self.get_avg_loss(test_loader)
                
                if track:
                    wandb.log({"val_loss": val_loss})
                    
                    
                print(f'Validation loss: {val_loss}')
                if val_loss < best_loss:
                    best_loss = val_loss
                    self.save_model(save_as)
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
                    if no_improve_epochs >= patience:
                        print("Early stopping")
                        return


    def get_avg_loss(self, data_loader):
        self.to(self.device)
        self.eval()
        total_loss = 0
        loss_calculated = 0
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                loss_calculated += 1

                
        avg_loss = total_loss / loss_calculated
        return avg_loss


    #TODO: Fix evaluation function and put it in a separate file
    def evaluate(self, test_loader):
        self.to(self.device)
        self.eval()
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.to_segmentation(data)

                pixel_accuracy, mean_iou = evaluate_model(output, target, return_values=True, print_values=False)
                print(f'Pixel accuracy: {pixel_accuracy.item()}, Mean IoU: {mean_iou.item()}')
                
    def save_model(self, fileName):
        torch.save(self.state_dict(), f"saved_models/{fileName}")
        
    def load_model(self, file_path, map_location = None):
        if map_location is None:
            self.load_state_dict(torch.load(f"saved_models/{file_path}"))
            
        else:
            self.load_state_dict(torch.load(f"saved_models/{file_path}"), map_location=map_location)
    
if __name__ == "__main__":
    BATCH_SIZE = 15
    TRAIN_SIZE = 0.8
    SQUARE_SIZE = 256
    EPOCHS = 11
    N_NEURONS = 64
    LEARNING_RATE = 0.001
    PATIENCE = 5
    N_DEPTH = 3

    IN_MEMORY = False
    STATIC_TEST = True
    WITH_SKIP_CONNECTIONS = True
    
    save_as = "best_model.pth"
    
    train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE, train_size=TRAIN_SIZE, square_size=SQUARE_SIZE, in_memory=IN_MEMORY, static_test=STATIC_TEST)

    model = UNet2D(n_neurons=N_NEURONS,
                    n_channels=2,
                    n_classes=3,
                    n_depth=N_DEPTH,
                    with_skip_connections=WITH_SKIP_CONNECTIONS,
                    save_as=save_as)
    
    # model.train_model(train_loader = train_loader, test_loader = test_loader, epochs=EPOCHS, lr=LEARNING_RATE)
    model.train_model(train_loader = train_loader, test_loader=test_loader, epochs=EPOCHS, lr=LEARNING_RATE, patience=PATIENCE)
    
    model.evaluate(test_loader)
    
    print("done")
    