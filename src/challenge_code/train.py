import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from .dataset import ChallengeImageFolderDataset
from .model import ConvolutionalNetwork
from .model import VGG19Timm
from .model import DinoV2Timm
from .model import VGG19BN


def get_device(force: str = "auto") -> torch.device:
    force = force.lower()
    if force == "cpu":
        return torch.device("cpu")
    if force == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_output_folder() -> Path:
    base = Path(__file__).parent / "outs"
    base.mkdir(parents=True, exist_ok=True)

    i = 1
    while True:
        out = base / f"run_{i:03d}"
        if not out.exists():
            out.mkdir(parents=True, exist_ok=True)
            return out
        i += 1


@torch.no_grad()
def validate(loader, model, criterion, device, pin_memory: bool):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=pin_memory)
        y = y.to(device, non_blocking=pin_memory)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / max(1, len(loader))
    acc = correct / max(1, total)
    return avg_loss, acc


def train_model():
    torch.manual_seed(42)

    # === CONFIG ===
    dataset_root = Path(r"C:\TAIA\challenge_grupo_2_2026\dataset_challenge")
    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"

    num_classes = 19
    epochs = 20
    lr = 5e-6
    batch_size = 32

    # === EARLY STOPPING CONFIG ===
    patience = 4          # cuántas épocas esperar sin mejora
    min_delta = 0.0       # mejora mínima requerida en val_loss para resetear paciencia
    # ====================================

    device = get_device("auto")
    out_dir = make_output_folder()

    print("Device:", device)
    print("Train dir:", train_dir)
    print("Val dir:", val_dir)
    print("Output dir:", out_dir)

    if not train_dir.exists():
        raise FileNotFoundError(f"No existe: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"No existe: {val_dir}")

    # Transforms
    imgsize = 224
    transform = transforms.Compose(
        [
            transforms.Resize((imgsize, imgsize)),
            transforms.ToTensor(),
            transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        ]
    )
    transform_train = transforms.Compose(
        [
            transforms.Resize((imgsize, imgsize)),
            transforms.RandomApply(
                [
                    transforms.RandomRotation(45),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
                ],
                p=0.8,
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        ]
    )

    # Datasets
    train_dataset = ChallengeImageFolderDataset(train_dir, transform=transform_train)
    val_dataset = ChallengeImageFolderDataset(val_dir, transform=transform)

    print("Train clases detectadas:", train_dataset.classes)
    print("Train #clases:", len(train_dataset.classes), "| Train imgs:", len(train_dataset))

    print("Val clases detectadas:", val_dataset.classes)
    print("Val #clases:", len(val_dataset.classes), "| Val imgs:", len(val_dataset))

    # === DataLoaders ===
    pin_memory = (device.type == "cuda")
    num_workers = 8

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    # === Model ===
    # model = ConvolutionalNetwork(num_classes).to(device)
    #model = VGG19Timm(num_classes=19, pretrained=True).to(device)
    model = DinoV2Timm(num_classes=19, pretrained=True).to(device)
    #model = VGG19BN(num_classes=19, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_path = out_dir / "best_model.pth"

    train_losses = []
    val_losses = []
    val_accs = []


    no_improve_epochs = 0
  

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=True)
        for x, y in pbar:
            x = x.to(device, non_blocking=pin_memory)
            y = y.to(device, non_blocking=pin_memory)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()
            pbar.set_postfix(loss=loss.item())

        train_loss = running / max(1, len(train_loader))
        train_losses.append(train_loss)

        val_loss, val_acc = validate(val_loader, model, criterion, device, pin_memory)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # === EARLY STOPPING LOGIC ===
        improved = (best_val_loss - val_loss) > min_delta
        if improved:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
        # ====================================

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
            f"Best Val Loss: {best_val_loss:.4f}"
        )

        # === STOP CONDITION  ===
        if no_improve_epochs >= patience:
            print(
                f"\nEarly stopping activado: no hay mejora en Val Loss por {patience} épocas. "
                f"Mejor Val Loss: {best_val_loss:.4f}"
            )
            break
        # ===============================

    print("\nGuardado mejor modelo en:", best_path)

        # === Plots ===
    ran_epochs = len(train_losses)  

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, ran_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, ran_epochs + 1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_plot.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, ran_epochs + 1), val_accs, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "val_acc_plot.png")
    plt.close()

    # === Metadata ===
    (out_dir / "info.txt").write_text(
        "\n".join(
            [
                f"train_dir={train_dir}",
                f"val_dir={val_dir}",
                f"num_classes={num_classes}",
                f"TRAIN_IMAGES={len(train_dataset)}",
                f"VAL_IMAGES={len(val_dataset)}",
                f"IMG_SIZE={imgsize}x{imgsize}",
                f"epochs={epochs}",
                f"batch_size={batch_size}",
                f"lr={lr}",
                f"NUM_WORKERS={num_workers}",
                f"BEST_VAL_LOSS={best_val_loss}",
                f"BEST_MODEL={best_path}",
                f"CLASS_TO_IDX={train_dataset.class_to_idx}",
                f"Model_Used={model.__class__.__name__}",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    train_model()