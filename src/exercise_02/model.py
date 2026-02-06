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
    def __init__(self, input_dim, output_dim, num_hidden_neurons1, num_hidden_neurons2, apodo):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, num_hidden_neurons1)
        self.fc2 = nn.Linear(num_hidden_neurons1, num_hidden_neurons2)
        self.fc3 = nn.Linear(num_hidden_neurons2, output_dim)
        self.activation1 = nn.LeakyReLU()
        self.activation2 = nn.LeakyReLU()
        self.activation3 = nn.Identity()
        self.apodo = apodo

    def forward(self, x, use_activation=True):
        x1 = self.fc1(x)

        x1 = self.activation1(x1)

        x2 = self.fc2(x1)

        x2 = self.activation2(x2)

        x3 = self.fc3(x2)

        if use_activation:
            x3 = self.activation3(x3)
        return x3


if __name__ == "__main__":
    model2 = MultilayerPerceptron(1000, 2, 256, 256, "mi_modelo_de_desfibrilador")

    x = torch.tensor([1.0])
    print(model2.forward(x))
    pass
    # x = torch.tensor([1.0])
    # print(model(x))
    # pass
