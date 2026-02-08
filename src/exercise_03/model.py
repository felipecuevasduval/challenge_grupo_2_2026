import torch
import torch.nn as nn


class SimplePerceptron(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.activation = nn.Identity()

    def forward(self, x, use_activation=True):
        x = self.fc(x)
        if use_activation:
            x = self.activation(x)
        return x


class MultilayerPerceptron(nn.Module):
    def __init__(self, input_dim, output_dim, num_hidden_neurons, apodo=None):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, num_hidden_neurons)
        self.fc2 = nn.Linear(num_hidden_neurons, num_hidden_neurons)
        self.fc3 = nn.Linear(num_hidden_neurons, num_hidden_neurons)
        self.fc4 = nn.Linear(num_hidden_neurons, output_dim)  # (Mejora) 3 capas totales: 2 ReLU + salida

        self.relu = nn.ReLU()
        self.out_act = nn.Identity()
        self.apodo = apodo

    def forward(self, x, use_activation=True):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        if use_activation:
            x = self.out_act(x)
        return x

if __name__ == "__main__":
    model2 = MultilayerPerceptron(1000, 2, 256, "mi_modelo_de_desfibrilador")

    x = torch.tensor([1.0])
    print(model2.forward(x))
    pass
