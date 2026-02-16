import torch
import torch.nn as nn


class ConvolutionalNetwork(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        #Arquitectura inspirada en VGG.
        # Conv-ReLU-MaxPooling  (32x32 -> 16x16)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv-ReLU-MaxPooling (16x16 -> 8x8)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Flattening + FCs
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.relu3 = nn.ReLU()

        # Se añade Dropout porque sin el se veia overfitting:
        # el train loss bajaba pero el val loss se mantenia alto o subia.
        self.dropout = nn.Dropout(p=0.4)

        self.fc2 = nn.Linear(512, 512)
        self.relu4 = nn.ReLU() # Declaramos cada ReLu por separado por si queremos 
                               #usar una activacion diferente en cada capa
        # Salida para clasificación
        self.fc_out = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        
        x = self.pool2(self.relu2(self.conv2(x)))

        x = self.flatten(x)

        x = self.relu3(self.fc1(x))
        x = self.dropout(x)

        x = self.relu4(self.fc2(x))
        x = self.fc_out(x)

        return x


if __name__ == "__main__":
    model = ConvolutionalNetwork(10)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print("Output shape:", y.shape)  # [2, 10]
