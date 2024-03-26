
from dataLoad import data_load_tensors, get_dataloaders
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_device
from evaluation import evaluate_model
import numpy as np

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet2D(nn.Module):
    def __init__(self, n_neurons, n_channels, n_classes, padding=False):
        super().__init__()
        self.encoder = nn.Sequential(
            DoubleConv(n_channels, n_neurons),
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(n_neurons, n_neurons * 2),
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(n_neurons * 2, n_neurons * 4),
            nn.MaxPool2d(kernel_size=2),
        )

        self.decoder = nn.Sequential(
            DoubleConv(n_neurons * 4, n_neurons * 2),
            nn.ConvTranspose2d(n_neurons * 2, n_neurons * 2, kernel_size=2, stride=2),
            DoubleConv(n_neurons * 2, n_neurons),
            nn.ConvTranspose2d(n_neurons, n_neurons, kernel_size=2, stride=2),
            DoubleConv(n_neurons, n_neurons),
            nn.ConvTranspose2d(n_neurons, n_neurons, kernel_size=2, stride=2),
            DoubleConv(n_neurons, n_classes),
        )
        self.n_classes = n_classes
        self.padding = padding
        self.criterion = None

        self.optimizer = None
        self.device = get_device()

        self.final_conv = nn.Conv2d(in_channels=self.n_classes, out_channels=self.n_classes, kernel_size=1)

    def forward(self, x):
        # Pad the input to a size divisible by 8
        if self.padding:
            _, _, h, w = x.shape
            pad_h = (h + 7) // 8 * 8 - h
            pad_w = (w + 7) // 8 * 8 - w
            x = F.pad(x, (0, pad_w, 0, pad_h))

        x = self.encoder(x)
        x = self.decoder(x)
        x = self.final_conv(x)

        if self.padding:
            x = x[:, :, :h, :w]
        return x
    
    def to_segmentation(self, x):
        return x.argmax(dim=1)

    def train_model(self, train_loader, test_loader = None, optimizer = "adam", lr = 0.001
                    ,criterion = "crossentropy", epochs = 10, verbose=2, patience=5):
        
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

            if test_loader:
                val_loss = self.get_avg_loss(test_loader)
                print(f'Validation loss: {val_loss}')
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(self.state_dict(), 'best_model.pth')
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
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.to_segmentation(data)

                pixel_accuracy, mean_iou = evaluate_model(output, target, return_values=True, print_values=False)
                print(f'Pixel accuracy: {pixel_accuracy.item()}, Mean IoU: {mean_iou.item()}')
    

if __name__ == "__main__":
    BATCH_SIZE = 30
    TRAIN_SIZE = 0.8
    SQUARE_SIZE = 256
    EPOCHS = 15
    N_NEURONS = 64
    LEARNING_RATE = 0.001
    PATIENCE = 5

    IN_MEMORY = False
    STATIC_TEST = True
    train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE, train_size=TRAIN_SIZE, square_size=SQUARE_SIZE, in_memory=IN_MEMORY, static_test=STATIC_TEST)

    model = UNet2D(n_neurons=N_NEURONS,
                    n_channels=2,
                    n_classes=3)
    
    # model.train_model(train_loader = train_loader, test_loader = test_loader, epochs=EPOCHS, lr=LEARNING_RATE)
    model.train_model(train_loader = train_loader, test_loader=test_loader, epochs=EPOCHS, lr=LEARNING_RATE, patience=PATIENCE)
    
    model.evaluate(test_loader)
    
    print("done")
    