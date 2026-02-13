import torch
import torch.nn as nn


class ConvolutionalNetwork(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.layer1 = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU())
        self.layer2 = nn.Sequential(
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(), 
        nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(7*7*64, 512),
        nn.ReLU())
        self.fc1 = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(512, 512),
        nn.ReLU())
        self.fc2= nn.Sequential(
        nn.Linear(512, num_classes))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


if __name__ == "__main__":
    model = ConvolutionalNetwork(10, 2)

    x = torch.tensor([1.0])
    print(model.forward(x))
    pass
