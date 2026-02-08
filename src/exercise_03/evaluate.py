from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader, random_split

from .dataset import NoisyRegressionDataset
from .model import MultilayerPerceptron


def evaluate_and_plot(loader, model, dataset_name, output_folder):
    model.eval()
    device = next(model.parameters()).device  # (Mejora) así evitamos el típico error de CPU vs CUDA en evaluación
    all_inputs = []
    all_outputs = []
    all_targets = []

    # (Mejora) si el dataset normaliza x, esto nos permite volver a escala original para los plots
    base_dataset = getattr(loader.dataset, "dataset", loader.dataset)
    x_scale = getattr(base_dataset, "x_scale", 1.0)

    with torch.no_grad():
        for inputs, targets in loader:
            # (Mejora) non_blocking aprovecha pin_memory del DataLoader cuando estás en CUDA
            inputs = inputs.to(device, non_blocking=True)  # (Mejora) inputs y modelo siempre en el mismo device
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs, use_activation=False)

            # (Mejora) para pasar a numpy, primero movemos a CPU (en CUDA, .numpy() directo da error)
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

    # Calculate r2, MAE and MSE
    r2 = 1 - np.sum((all_targets - all_outputs) ** 2) / np.sum(
        (all_targets - np.mean(all_targets)) ** 2
    )
    mae = np.mean(np.abs(all_targets - all_outputs))
    mse = np.mean((all_targets - all_outputs) ** 2)

    metrics = {
        "R2": r2,
        "MAE": mae,
        "MSE": mse,
    }

    print(f"Evaluation metrics for {dataset_name} dataset:")
    print(metrics)

    ax = sns.regplot(df, x="y_true", y="y_pred", label=dataset_name)
    ax.set_title(f"Regression plot for {dataset_name} dataset")
    plt.legend()
    plt.savefig(f"{output_folder}/{dataset_name}_regression_plot.png")
    plt.show()
    plt.close()

    # Plot the data points
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="x", y="y_true", label="True")
    sns.scatterplot(data=df, x="x", y="y_pred", label="Predicted")
    plt.xlabel("x")
    plt.ylabel("y")  # Adding y label
    plt.title(f"Data points for {dataset_name} dataset")
    plt.legend()
    plt.savefig(f"{output_folder}/{dataset_name}_data_points_plot.png")
    plt.show()
    plt.close()

    return metrics


def save_metrics_as_picture(metrics, filepath):
    # Create a DataFrame
    df = pd.DataFrame(metrics)

    # Round the values to 3 decimal places
    df = df.round(3)

    # Plot the table and save as an image
    fig, ax = plt.subplots(figsize=(8, 2))  # set size frame
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(
        cellText=df.values, colLabels=df.columns, rowLabels=df.index, cellLoc="center", loc="center"
    )

    # Save the plot as an image
    plt.savefig(filepath)


if __name__ == "__main__":
    output_folder = Path(__file__).parent.parent.parent / "outs" / Path(__file__).parent.name
    output_folder.mkdir(exist_ok=True, parents=True)
    # Set the seed for reproducibility
    torch.manual_seed(42)
    # Create an instance of the dataset
    dataset = NoisyRegressionDataset(size=10000)

    # Split the dataset into train, validation, and test sets
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Create DataLoaders for the datasets
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # (Mejora) usamos CUDA si está disponible
    pin_memory = True if device.type == "cuda" else False  # (Mejora) acelera copias a GPU
    batch_size = 256  # (Mejora) consistente con train.py

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    # Load the best model weights
    model = MultilayerPerceptron(
        input_dim=1,
        output_dim=1,
        num_hidden_neurons=256,  # consistente con train.py
        apodo="modelo",
    ).to(device)
    model.load_state_dict(torch.load(output_folder / "best_model.pth", map_location=device))  # (Mejora) sin lios CPU/CUDA

    metrics = {}
    # Evaluate and plot for train, validation and test datasets
    metrics["train"] = evaluate_and_plot(train_loader, model, "train", output_folder)
    metrics["validation"] = evaluate_and_plot(val_loader, model, "validation", output_folder)
    metrics["test"] = evaluate_and_plot(test_loader, model, "test", output_folder)

    # save  metrics as csv
    pd.DataFrame(metrics).to_csv(output_folder / "metrics.csv")

    # Save the metrics as an image
    save_metrics_as_picture(metrics, output_folder / "metrics.png")

    print("Evaluation complete!")
