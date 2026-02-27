from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def count_images_by_class(split_dir: Path) -> Counter:
    """
    Cuenta imágenes por subcarpeta (clase) dentro de split_dir.
    Estructura esperada: split_dir/<clase>/*.jpg|png...
    """
    counts = Counter()
    if not split_dir.exists():
        raise FileNotFoundError(f"No existe: {split_dir}")

    for class_dir in split_dir.iterdir():
        if not class_dir.is_dir():
            continue
        n = 0
        for p in class_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                n += 1
        counts[class_dir.name] = n

    # Si hay clases con 0, las deja; si no hay subcarpetas, avisa
    if not counts:
        raise RuntimeError(f"No encontré subcarpetas de clases en: {split_dir}")
    return counts


def plot_histogram(counts: Counter, title: str, save_path: Path | None = None, rotate_xticks: int = 0):
    classes = sorted(counts.keys())
    values = [counts[c] for c in classes]

    plt.figure(figsize=(12, 5))
    plt.bar(classes, values)
    plt.title(title)
    plt.xlabel("Clase")
    plt.ylabel("# Imágenes")
    if rotate_xticks:
        plt.xticks(rotation=rotate_xticks, ha="right")
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
    plt.show()
    plt.close()


def merge_counts(*counters: Counter) -> Counter:
    total = Counter()
    for c in counters:
        total.update(c)
    return total


def main():
    # AJUSTA ESTA RUTA BASE
    dataset_root = Path(r"C:\TAIA\challenge_grupo_2_2026\dataset_challenge")

    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"
    test_dir = dataset_root / "test"   # o "test3" si quieres analizar el augmentado
    out_dir = Path(__file__).parent / "outs" / "_histograms"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_counts = count_images_by_class(train_dir)
    val_counts = count_images_by_class(val_dir)
    test_counts = count_images_by_class(test_dir)

    # Hist por separado
    plot_histogram(train_counts, "Distribución de clases - TRAIN", out_dir / "hist_train.png", rotate_xticks=0)
    plot_histogram(val_counts, "Distribución de clases - VAL", out_dir / "hist_val.png", rotate_xticks=0)
    plot_histogram(test_counts, "Distribución de clases - TEST", out_dir / "hist_test.png", rotate_xticks=0)

    # Hist todo junto (suma train+val+test)
    all_counts = merge_counts(train_counts, val_counts, test_counts)
    plot_histogram(all_counts, "Distribución de clases - TOTAL (train+val+test)", out_dir / "hist_all.png", rotate_xticks=0)

    print("Listo. Guardé los PNG en:", out_dir)


if __name__ == "__main__":
    main()