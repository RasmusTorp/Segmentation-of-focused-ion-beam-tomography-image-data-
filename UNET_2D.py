
from dataLoad import data_load_tensors, get_dataloaders
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, n_neurons):
        super().__init__()
        self.encoder = nn.Sequential(
            DoubleConv(2, n_neurons),
            nn.MaxPool2d(2),
            DoubleConv(n_neurons, n_neurons * 2),
            nn.MaxPool2d(2),
            DoubleConv(n_neurons * 2, n_neurons * 4),
            nn.MaxPool2d(2),
        )

        self.decoder = nn.Sequential(
            DoubleConv(n_neurons * 4, n_neurons * 2),
            nn.ConvTranspose2d(n_neurons * 2, n_neurons * 2, kernel_size=2, stride=2),
            DoubleConv(n_neurons * 2, n_neurons),
            nn.ConvTranspose2d(n_neurons, n_neurons, kernel_size=2, stride=2),
            DoubleConv(n_neurons, n_neurons),
            nn.ConvTranspose2d(n_neurons, n_neurons, kernel_size=2, stride=2),
            DoubleConv(n_neurons, 3),
        )

        self.final_conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)

    def forward(self, x):
        # Pad the input to a size divisible by 8
        _, _, h, w = x.shape
        pad_h = (h + 7) // 8 * 8 - h
        pad_w = (w + 7) // 8 * 8 - w
        x = F.pad(x, (0, pad_w, 0, pad_h))

        x = self.encoder(x)
        x = self.decoder(x)
        x = self.final_conv(x)
        # x = x.squeeze(1)

        x = x[:, :, :h, :w]
        return x

    def train_model(self, device, train_loader, optimizer, criterion, epochs, verbose=2):
        self.to(device)
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = self(data)
                loss = criterion(output, target)

                if verbose == 2:
                    print(loss.item())

                loss.backward()
                optimizer.step()
            if verbose:
                print(f'Train Epoch: {epoch + 1}, Loss: {loss.item()}')


if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders()

    model = UNet2D(n_neurons=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    model.train_model(device, train_loader, optimizer, criterion, epochs=10)
    
    print("done")
    