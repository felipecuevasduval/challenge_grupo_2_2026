import torch
import torch.nn as nn


class ConvolutionalNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Conv2d(input_dim, output_dim, kernel_size=1)
        self.activation1 = nn.ReLU()
        self.fc2 = nn.Conv2d(output_dim, output_dim, kernel_size=1)
        self.activation2 = nn.ReLU()
        self.fc3 = nn.maxpool2d(kernel_size=2)
        self.activation3 = nn.Identity()

    def forward(self, x, use_activation=True):
        x = self.fc(x)
        if use_activation:
            x = self.activation(x)
        return x


if __name__ == "__main__":
    model = ConvolutionalNetwork(10, 2)

    x = torch.tensor([1.0])
    print(model.forward(x))
    pass
