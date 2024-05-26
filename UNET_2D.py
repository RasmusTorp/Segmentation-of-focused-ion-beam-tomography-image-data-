
from dataLoad import get_dataloaders
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_device
from evaluation.evaluation import evaluate_model
import numpy as np
import wandb
import matplotlib.pyplot as plt

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
                padding = 1, stride = 1, with_skip_connections=True, device = None):
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
        self.n_channels = n_channels
        self.criterion = None

        self.optimizer = None
        
        self.device = get_device() if not device else device

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
    
    # Method used to get the segmentation of the input
    def to_segmentation(self, x):
        x = self.forward(x)
        return x.argmax(dim=1)

    def train_model(self, train_loader, val_loader = None, optimizer = "adam", lr = 0.001
                    ,criterion = "crossentropy", epochs = 10, verbose=2, patience=5, save_as="best_model.pth", track = False):
        
        self.to(self.device)
    
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        if criterion == "crossentropy":
            self.criterion = nn.CrossEntropyLoss()
        
        best_loss = np.inf
        no_improve_epochs = 0
        
        if val_loader:
            val_loss = self.get_avg_loss(val_loader)
            print(f'Validation loss: {val_loss}')
            
            pixel_accuracy, mean_iou = self.evaluate(val_loader)
        
            if track:
                wandb.log({"val_loss": val_loss})
                wandb.log({"pixel_accuracy": pixel_accuracy.item()})
                wandb.log({"mean_iou": mean_iou.item()})

        for epoch in range(epochs):
            self.train()
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

            if val_loader:
                val_loss = self.get_avg_loss(val_loader)
                pixel_accuracy, mean_iou = self.evaluate(val_loader)
                print(f'Validation loss: {val_loss}')
                
                if track:
                    wandb.log({"val_loss": val_loss})
                    wandb.log({"pixel_accuracy": pixel_accuracy.item()})
                    wandb.log({"mean_iou": mean_iou.item()})
                    
                
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

    def evaluate(self, test_loader):
        self.to(self.device)
        self.eval()
        with torch.no_grad():
            all_outputs = []
            all_targets = []
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.to_segmentation(data)
                all_outputs.append(output)
                all_targets.append(target)

            all_outputs = torch.cat(all_outputs, dim=0)
            all_targets = torch.cat(all_targets, dim=0)

            pixel_accuracy, mean_iou = evaluate_model(all_outputs, all_targets, print_values=False)
            print(f'Pixel accuracy: {pixel_accuracy.item()}, Mean IoU: {mean_iou.item()}')
            return pixel_accuracy, mean_iou
            
                
    def save_model(self, fileName):
        torch.save(self.state_dict(), f"saved_models/{fileName}")
        
    def load_model(self, file_path, map_location = None):
        if map_location is None:
            state_dict = torch.load(file_path)
            self.load_state_dict(state_dict)
            
        else:
            state_dict = torch.load(file_path, map_location=map_location)

            self.load_state_dict(state_dict)

    # Method used exclusively for debugging and visualization for the report
    def plot_trough_network(self, x, save_as):
        self.eval()
        encoder_outs = []
        
        x_original = x
        
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        detector1 = x[0][0].detach().numpy()
        detector2 = x[0][1].detach().numpy()
        
        axs[0].imshow(detector1)
        axs[0].set_title('Detector 1')
        
        axs[1].imshow(detector2)
        axs[1].set_title('Detector 2')
        
        plt.tight_layout()
        plt.savefig(f"plots/{save_as}_input.png")
        plt.clf()
        
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            plt.figure(figsize=(9, 7))
            plt.imshow(x[0][0].detach().numpy())
            plt.title(f'Encoder Layer {i+1}, Shape: {list(x.shape)}')
            plt.tight_layout()
            plt.savefig(f"plots/{save_as}_{i}.png")
            plt.clf()
            encoder_outs.append(x)
            
        for i, layer in enumerate(self.decoder):
            x = layer(x, encoder_outs[-i-2])
            plt.figure(figsize=(9, 7))
            plt.imshow(x[0][0].detach().numpy())
            plt.title(f'Decoder Layer {i+1}, Shape: {list(x.shape)}')
            plt.tight_layout()
            plt.savefig(f"plots/{save_as}_{len(self.encoder) + i}.png")
            plt.clf()
            
        x = self.final_conv(x)
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        plt.title(f'Final Convolution, Shape: {list(x.shape)}')
        axs[0].imshow(x[0][0].detach().numpy())
        axs[0].set_title(f'P(X = 0)')
        axs[1].imshow(x[0][1].detach().numpy())
        axs[1].set_title(f'P(X = 1)')
        axs[2].imshow(x[0][2].detach().numpy())
        axs[2].set_title(f'P(X = 2)')
        plt.tight_layout()
        plt.savefig(f"plots/{save_as}_final.png")
        plt.clf()
        
        encoder_outs = []
        
    
        # In one plot
        x = x_original
        encoder_outs = []
        
        fig, axs = plt.subplots(4, 2, figsize=(10, 20))  # Create a figure and a 4x2 subplot grid

        for i, layer in enumerate(self.encoder):
            x = layer(x)
            axs[i//2, i%2].imshow(x[0][0].detach().numpy())
            axs[i//2, i%2].set_title(f'Encoder Layer {i+1}, Shape: {list(x.shape)}')
            encoder_outs.append(x)
            
        for i, layer in enumerate(self.decoder):
            x = layer(x, encoder_outs[-i-2])
            axs[(i+len(self.encoder))//2, (i+len(self.encoder))%2].imshow(x[0][0].detach().numpy())
            axs[(i+len(self.encoder))//2, (i+len(self.encoder))%2].set_title(f'Decoder Layer {i+1}, Shape: {list(x.shape)}')
            
        x = self.final_conv(x)
        axs[-1, -1].imshow(x[0][0].detach().numpy())
        axs[-1, -1].set_title(f'Final Convolution, Shape: {list(x.shape)}')

        plt.subplots_adjust(hspace=-0.3) 
        plt.tight_layout()
        plt.savefig(f"plots/{save_as}.png")
        plt.clf()
        
        segmentation = x.argmax(dim=1)
        plt.figure(figsize=(9, 7))
        plt.imshow(segmentation[0].detach().numpy())
        plt.title('Segmentation')
        plt.tight_layout()
        plt.savefig(f"plots/{save_as}_segmentation.png")
        plt.clf()
        
    
if __name__ == "__main__":
    BATCH_SIZE = 15
    TRAIN_SIZE = 0.8
    SAMPLING_HEIGHT= 256
    SAMPLING_WIDTH = 256*2
    EPOCHS = 2
    N_NEURONS = 64
    LEARNING_RATE = 0.001
    PATIENCE = 5
    N_DEPTH = 3

    IN_MEMORY = True
    STATIC_TEST = True
    WITH_SKIP_CONNECTIONS = True
    RANDOM_TRAIN_TEST_SPLIT = True
    
    
    DETECTOR = "both"
    
    n_channels = 2 if DETECTOR == "both" else 1
    save_as = "best_model.pth"
    
    train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE, train_size=TRAIN_SIZE, 
                                                sampling_height=SAMPLING_HEIGHT, sampling_width=SAMPLING_WIDTH, 
                                                in_memory=IN_MEMORY, 
                                                static_test=STATIC_TEST, detector=DETECTOR,
                                                random_train_test_split=RANDOM_TRAIN_TEST_SPLIT)

    model = UNet2D(n_neurons=N_NEURONS,
                    n_channels=n_channels,
                    n_classes=3,
                    n_depth=N_DEPTH,
                    with_skip_connections=WITH_SKIP_CONNECTIONS)
    
    model.evaluate(test_loader)
    
    model.train_model(train_loader = train_loader, test_loader=test_loader, 
                    epochs=EPOCHS, lr=LEARNING_RATE, patience=PATIENCE,save_as=save_as)
    
    model.evaluate(test_loader)
    
    print("done")
    