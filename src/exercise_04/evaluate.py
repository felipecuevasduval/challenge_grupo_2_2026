from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix  

from .dataset import CIFAR10Dataset
from .model import ConvolutionalNetwork


def get_device(force: str = "auto") -> torch.device:
    force = force.lower()
    if force == "cpu":
        return torch.device("cpu")
    if force == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_classification(loader, model, criterion):
    #Devuelve loss medio y accuracy.
    model.eval()
    device = next(model.parameters()).device

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)  # logits [N, 10]
            loss = criterion(logits, targets)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)

            correct += (preds == targets).sum().item()
            total += targets.size(0)

    avg_loss = total_loss / max(1, len(loader))
    acc = correct / max(1, total)
    return avg_loss, acc


def confusion_matrix_and_plot(loader, model, dataset_name, output_folder, num_classes=10):
    model.eval()
    device = next(model.parameters()).device

    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]

    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            preds = torch.argmax(logits, dim=1)

            y_true.append(targets.detach().cpu().numpy())
            y_pred.append(preds.detach().cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    # Plot con nombres + números
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        square=True,
        xticklabels=class_names,
        yticklabels=class_names,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion Matrix - {dataset_name}")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_folder / f"{dataset_name}_confusion_matrix.png")
    plt.show()
    plt.close()

    return cm



def save_metrics_as_picture(metrics, filepath):
    df = pd.DataFrame(metrics).round(3)

    fig, ax = plt.subplots(figsize=(8, 2))
    ax.axis("tight")
    ax.axis("off")
    ax.table(
        cellText=df.values,
        colLabels=df.columns,
        rowLabels=df.index,
        cellLoc="center",
        loc="center",
    )
    plt.savefig(filepath)
    plt.close()


def evaluate_and_plot(loader, model, dataset_name, output_folder):
    model.eval()
    device = next(model.parameters()).device
    all_inputs = []
    all_outputs = []
    all_targets = []

    base_dataset = getattr(loader.dataset, "dataset", loader.dataset)
    x_scale = getattr(base_dataset, "x_scale", 1.0)

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)  

            all_inputs.append((inputs.detach().cpu().numpy()) * x_scale)
            all_outputs.append(outputs.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())

    all_inputs = np.concatenate(all_inputs)
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)

    df = pd.DataFrame(
        data=np.array(
            [all_inputs.flatten(), all_targets.flatten(), all_outputs.flatten()]
        ).transpose(),
        columns=["x", "y_true", "y_pred"],
    )

    r2 = 1 - np.sum((all_targets - all_outputs) ** 2) / np.sum(
        (all_targets - np.mean(all_targets)) ** 2
    )
    mae = np.mean(np.abs(all_targets - all_outputs))
    mse = np.mean((all_targets - all_outputs) ** 2)

    metrics = {"R2": r2, "MAE": mae, "MSE": mse}

    print(f"Evaluation metrics for {dataset_name} dataset:")
    print(metrics)

    ax = sns.regplot(df, x="y_true", y="y_pred", label=dataset_name)
    ax.set_title(f"Regression plot for {dataset_name} dataset")
    plt.legend()
    plt.savefig(f"{output_folder}/{dataset_name}_regression_plot.png")
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="x", y="y_true", label="True")
    sns.scatterplot(data=df, x="x", y="y_pred", label="Predicted")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Data points for {dataset_name} dataset")
    plt.legend()
    plt.savefig(f"{output_folder}/{dataset_name}_data_points_plot.png")
    plt.show()
    plt.close()

    return metrics


if __name__ == "__main__":
    torch.manual_seed(42)

    output_folder = Path(__file__).parent.parent.parent / "outs" / Path(__file__).parent.name
    output_folder.mkdir(exist_ok=True, parents=True)

    device = get_device("auto")
    print(f"Using device: {device}")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))]
    )

    # Datasets/Loaders
    train_dataset = CIFAR10Dataset("./data", train=True, transform=transform)
    test_dataset = CIFAR10Dataset("./data", train=False, transform=transform)

    batch_size = 128
    pin_memory = True if device.type == "cuda" else False

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    # Modelo y pesos
    model = ConvolutionalNetwork(num_classes=10).to(device)
    model.load_state_dict(torch.load(output_folder / "best_model.pth", map_location=device))

    criterion = nn.CrossEntropyLoss()

    # Métricas de clasificación
    metrics = {}
    train_loss, train_acc = evaluate_classification(train_loader, model, criterion)
    test_loss, test_acc = evaluate_classification(test_loader, model, criterion)

    metrics["train"] = {"loss": train_loss, "accuracy": train_acc}
    metrics["test"] = {"loss": test_loss, "accuracy": test_acc}

    print("Classification metrics:")
    print(metrics)

    # Guardar métricas
    pd.DataFrame(metrics).to_csv(output_folder / "metrics.csv")
    save_metrics_as_picture(metrics, output_folder / "metrics.png")

    # Matriz de confusión con números (y CSV)
    confusion_matrix_and_plot(train_loader, model, "train", output_folder, num_classes=10)
    confusion_matrix_and_plot(test_loader, model, "test", output_folder, num_classes=10)

    print("Evaluation complete!")
