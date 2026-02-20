from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms

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
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(output_folder: Path, device: torch.device):
    # Data augmentation (NO CAMBIAR: tu transform)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))]
    )

    # Datasets
    train_dataset = CIFAR10Dataset("./data", train=True, transform=transform)
    val_dataset = CIFAR10Dataset("./data", train=False, transform=transform)
    test_dataset = CIFAR10Dataset("./data", train=False, transform=transform)

    # DataLoaders
    batch_size = 128
    pin_memory = True if device.type == "cuda" else False
    # Aunmentamos el numero de workers para aprovechar mejor la GPU.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory
                            , num_workers=12, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, 
                            num_workers=12, persistent_workers=True)
    #test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    # Model, loss, optimizer
    num_classes = 10
    model = ConvolutionalNetwork(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001) #Bajamos el lr de 0.001 a 0.0001 para 
                                                           # tener un entrenamiento que converja mejor.
    # Training loop
    num_epochs = 35
    best_val_loss = float("inf")
    best_model_path = output_folder / "best_model.pth"

    train_losses = []
    val_losses = []

    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            # aprovecha mejor la gpu con non_blocking=True (si el device es cuda), pero no es necesario en cpu.
            inputs = inputs.to(device, non_blocking=pin_memory)
            targets = targets.to(device, non_blocking=pin_memory)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                # (3) non_blocking=True
                inputs = inputs.to(device, non_blocking=pin_memory)
                targets = targets.to(device, non_blocking=pin_memory)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}"
            )

    print(f"Best validation loss: {best_val_loss:.4f}, Model saved to {best_model_path}")

    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(num_epochs), train_losses, label="Train Loss")
    plt.plot(range(num_epochs), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.savefig(output_folder / "loss_plot.png")


if __name__ == "__main__":
    torch.manual_seed(42)

    output_folder = Path(__file__).parent.parent.parent / "outs" / Path(__file__).parent.name
    output_folder.mkdir(exist_ok=True, parents=True)

    device = get_device("auto")
    print(f"Using device: {device}")

    train_model(output_folder, device=device)
