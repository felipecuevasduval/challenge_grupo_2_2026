from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

from .dataset import ChallengeImageFolderDataset
from .model import ConvolutionalNetwork
from .model import VGG19Timm


def get_device(force: str = "auto") -> torch.device:
    force = force.lower()
    if force == "cpu":
        return torch.device("cpu")
    if force == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def collect_predictions(loader, model):
    model.eval()
    device = next(model.parameters()).device

    y_true = []
    y_pred = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)
        preds = torch.argmax(logits, dim=1)

        y_true.append(targets.detach().cpu().numpy())
        y_pred.append(preds.detach().cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    return y_true, y_pred


def compute_ovr_metrics_from_cm(cm: np.ndarray):
    """
    Para multi-clase: calcula métricas por clase con One-vs-Rest y promedia (macro).
    Devuelve:
      sensitivity (recall), specificity, ppv (precision), npv
    """
    num_classes = cm.shape[0]

    sens = []
    spec = []
    ppv = []
    npv = []

    total = cm.sum()

    for k in range(num_classes):
        TP = cm[k, k]
        FN = cm[k, :].sum() - TP
        FP = cm[:, k].sum() - TP
        TN = total - TP - FN - FP

        # Sensitivity / Recall: TP / (TP + FN)
        sens_k = TP / (TP + FN) if (TP + FN) > 0 else np.nan

        # Specificity: TN / (TN + FP)
        spec_k = TN / (TN + FP) if (TN + FP) > 0 else np.nan

        # PPV / Precision: TP / (TP + FP)
        ppv_k = TP / (TP + FP) if (TP + FP) > 0 else np.nan

        # NPV: TN / (TN + FN)
        npv_k = TN / (TN + FN) if (TN + FN) > 0 else np.nan

        sens.append(sens_k)
        spec.append(spec_k)
        ppv.append(ppv_k)
        npv.append(npv_k)

    # macro mean ignorando NaN (por si alguna clase no aparece)
    metrics = {
        "Sensitivity (macro)": float(np.nanmean(sens)),
        "Specificity (macro)": float(np.nanmean(spec)),
        "PPV / Precision (macro)": float(np.nanmean(ppv)),
        "NPV (macro)": float(np.nanmean(npv)),
    }

    # por clase
    per_class = pd.DataFrame(
        {
            "Sensitivity": sens,
            "Specificity": spec,
            "PPV": ppv,
            "NPV": npv,
        }
    )

    return metrics, per_class


def save_confusion_matrix(cm, class_names, out_png: Path, title: str):
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    # números encima
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, int(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def evaluate_classification_metrics(loader, model, num_classes: int):
    """
    Devolvemos:
      - confusion matrix
      - dict de métricas pedidas
      - df por clase con sens/spec/ppv/npv
    """
    y_true, y_pred = collect_predictions(loader, model)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    ovr_metrics, per_class = compute_ovr_metrics_from_cm(cm)

    metrics = {
        "Accuracy": float(acc),
        "F1-Score (macro)": float(f1_macro),
        "F1-Score (weighted)": float(f1_weighted),
        **ovr_metrics,
    }

    return cm, metrics, per_class


if __name__ == "__main__":
    torch.manual_seed(42)

    # === CONFIG ===
    dataset_root = Path(r"C:\TAIA\challenge_grupo_2_2026\dataset_challenge")
    test_dir = dataset_root / "test3"   # test set real
    num_classes = 19                 
    batch_size = 32

    # carpeta donde está best_model.pth
    run_dir = Path(__file__).parent / "outs" / "run_005"
    best_model = run_dir / "best_model.pth"

    out_dir = Path(__file__).parent / "outs" / "Prueba1"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device("auto")
    print("Using device:", device)
    print("Test dir:", test_dir)
    print("Model:", best_model)

    imgsize = 64
    # Transform  
    transform = transforms.Compose(
        [
            transforms.Resize((imgsize, imgsize)),
            transforms.ToTensor(),
            transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        ]
    )

    # === Dataset/Loader ===
    test_dataset = ChallengeImageFolderDataset(test_dir, transform=transform)

    if len(test_dataset.classes) != num_classes:
        raise ValueError(
            f"Se detectaron {len(test_dataset.classes)} clases en {test_dir} "
            f"({test_dataset.classes}) pero num_classes={num_classes}."
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True,
    )

    # === Modelo ===
    model = VGG19Timm(num_classes=num_classes, pretrained=True).to(device)
    model.load_state_dict(torch.load(best_model, map_location=device))
    model.eval()

    # === Evaluación ===
    cm, metrics, per_class = evaluate_classification_metrics(test_loader, model, num_classes)

    # === Imprimir como tú pediste ===
    print("\nConfussion matrix:")
    print(cm)

    print("\nMetrics over the test set:------")
    print(f"F1-Score (macro): {metrics['F1-Score (macro)']:.4f}")
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print(f"Sensitivity (macro): {metrics['Sensitivity (macro)']:.4f}")
    print(f"Especificity (macro): {metrics['Specificity (macro)']:.4f}")
    print(f"Negative predictive value (macro): {metrics['NPV (macro)']:.4f}")
    print(f"Positive predictive value (macro): {metrics['PPV / Precision (macro)']:.4f}")

    # === Guardar archivos ===
    # Confusion matrix csv + png
    cm_df = pd.DataFrame(cm, index=test_dataset.classes, columns=test_dataset.classes)
    cm_df.to_csv(out_dir / "confusion_matrix_test.csv")

    save_confusion_matrix(
        cm,
        class_names=test_dataset.classes,
        out_png=out_dir / "confusion_matrix_test.png",
        title="Confusion Matrix - TEST",
    )

    # Metrics csv
    pd.Series(metrics).to_csv(out_dir / "metrics_test.csv")

    # Per-class metrics
    per_class.index = test_dataset.classes
    per_class.to_csv(out_dir / "metrics_per_class_test.csv")

    print("\nEvaluation complete!")