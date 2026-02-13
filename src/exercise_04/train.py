from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .dataset import CIFAR10Dataset
from .model import ConvolutionalNetwork


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
    dataset = CIFAR10Dataset()

    # Split the dataset into train, validation, and test sets
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Create DataLoaders for the datasets
    pin_memory = True if device.type == "cuda" else False
    batch_size = 256  # (Mejora) batch un poco mas grande suele aprovechar mejor la GPU
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    # Define the model, loss function, and optimizer
    num_clases = 10
    model = ConvolutionalNetwork(num_clases).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.005)  # (Mejora) weight_decay mas suave

    # Training loop with validation and saving best weights
    num_epochs = 400  # (Mejora) 
    best_val_loss = float("inf")
    best_model_path = output_folder / "best_model.pth"

    train_losses = []
    val_losses = []

    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0  # (Mejora) lo acumulamos como tensor para evitar sincronizar GPU->CPU en cada batch
        for inputs, targets in train_loader:
            # Forward pass
            inputs_cuda = inputs.to(device, non_blocking=pin_memory)  # (Mejora) non_blocking + pin_memory acelera H2D
            targets_cuda = targets.to(device, non_blocking=pin_memory)
            outputs = model(inputs_cuda, use_activation=False)
            loss = criterion(outputs, targets_cuda)

            train_loss += loss.detach()

            # Backward pass and optimization
            optimizer.zero_grad(set_to_none=True)  # (Mejora) un poco mas eficiente en GPU
            loss.backward()
            optimizer.step()

        train_loss = (train_loss / len(train_loader)).item()
        train_losses.append(train_loss)

        # Validation step
        model.eval()
        val_loss = 0  # (Mejora) igual que train_loss, acumulamos sin .item() por batch
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs_cuda = inputs.to(device, non_blocking=pin_memory)
                targets_cuda = targets.to(device, non_blocking=pin_memory)
                outputs = model(inputs_cuda, use_activation=False)
                loss = criterion(outputs, targets_cuda)
                val_loss += loss.detach()

        val_loss = (val_loss / len(val_loader)).item()
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
