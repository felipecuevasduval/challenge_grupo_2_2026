import torch
import torch.nn as nn


class SimplePerceptron(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(input_dim, output_dim)
        self.activation = nn.Identity()

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        x = self.activation(x)
        return x


class MultilayerPerceptron(nn.Module):
    def __init__(self, input_dim, output_dim, num_hidden_neurons, apodo=None):
        super().__init__()
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(input_dim, num_hidden_neurons)
        self.fc2 = nn.Linear(num_hidden_neurons, num_hidden_neurons)
        self.fc3 = nn.Linear(num_hidden_neurons, num_hidden_neurons)
        self.fc4 = nn.Linear(num_hidden_neurons, output_dim)

        self.relu = nn.ReLU()
        self.apodo = apodo

    def forward(self, x):
        x = self.flatten(x)  # CIFAR10: [N, 3, 32, 32] -> [N, 3072]

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)  
        return x


if __name__ == "__main__":
    input_dim = 3 * 32 * 32
    output_dim = 10
    model2 = MultilayerPerceptron(input_dim, output_dim, 512, "mi_mlp")

    x = torch.randn(4, 3, 32, 32)
    y = model2(x)
    print("Output shape:", y.shape)  # [4, 10]
    pass
