from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .dataset import NoisyRegressionDataset
from .model import MultilayerPerceptron


def get_device(force: str = "auto") -> torch.device:
    """Return a torch.device based on the `force` option.

    force: 'auto'|'cpu'|'cuda' - when 'auto' will pick cuda if available.
    """
    force = force.lower()
    if force == "cpu":
        return torch.device("cpu")
    if force == "cuda":
        return torch.device("cuda")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(output_folder: Path, device: torch.device):
    # Create an instance of the dataset
    dataset = NoisyRegressionDataset(size=1000)

    # Split the dataset into train, validation, and test sets
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Create DataLoaders for the datasets
    pin_memory = True if device.type == "cuda" else False
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False, pin_memory=pin_memory)

    # Define the model, loss function, and optimizer
    input_dim = 1
    output_dim = 1
    num_hidden_neurons = 256
    apodo = "modelo_regresion_polinomio_grado_2"
    model = MultilayerPerceptron(
        input_dim, output_dim, num_hidden_neurons, num_hidden_neurons, apodo
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # Training loop with validation and saving best weights
    num_epochs = 200
    best_val_loss = float("inf")
    best_model_path = output_folder / "best_model.pth"

    train_losses = []
    val_losses = []

    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            # Forward pass
            inputs_cuda = inputs.to(device)
            targets_cuda = targets.to(device)
            outputs = model(inputs_cuda, use_activation=False)
            loss = criterion(outputs, targets_cuda)

            train_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs_cuda = inputs.to(device)
                targets_cuda = targets.to(device)
                outputs = model(inputs_cuda, use_activation=False)
                loss = criterion(outputs, targets_cuda)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}"
            )

    print(f"Best validation loss: {best_val_loss:.4f}, Model saved to {best_model_path}")

    # Plotting the training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(num_epochs), train_losses, label="Train Loss")
    plt.plot(range(num_epochs), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")

    # Save the plot to the outs/ folder
    plt.savefig(output_folder / "loss_plot.png")


if __name__ == "__main__":
    # Create output folder based on file folder
    output_folder = Path(__file__).parent.parent.parent / "outs" / Path(__file__).parent.name
    output_folder.mkdir(exist_ok=True, parents=True)

    device = get_device("auto")  # choices are "auto", "cpu", "cuda"
    print(f"Using device: {device}")

    # Set the seed for reproducibility
    torch.manual_seed(42)
    train_model(output_folder, device=device)
