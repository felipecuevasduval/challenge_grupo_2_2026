import os
import random
from pathlib import Path

from PIL import Image
from torchvision import transforms


def build_augmentation_pipeline(imgsize: int = 224):
    # Puedes ajustar estos valores si quieres más/menos agresivo
    return transforms.Compose(
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
        ]
    )


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def augment_folder(
    src_root: str,
    dst_root: str,
    num_aug_per_image: int = 3,
    imgsize: int = 224,
    seed: int = 42,
    copy_original: bool = True,
    out_ext: str = ".jpg",
    jpg_quality: int = 95,
):
    """
    Lee:  src_root/<clase>/*.jpg|png|...
    Escribe: dst_root/<clase>/
      - opcionalmente copia un 'original' normalizado (resize)
      - genera num_aug_per_image variaciones por imagen
    """

    random.seed(seed)

    src_root = Path(src_root)
    dst_root = Path(dst_root)

    if not src_root.exists():
        raise FileNotFoundError(f"No existe src_root: {src_root}")

    aug = build_augmentation_pipeline(imgsize=imgsize)

    # Detecta "clases" como subcarpetas directas (A, B, C, ...)
    class_dirs = [d for d in src_root.iterdir() if d.is_dir()]

    if not class_dirs:
        raise RuntimeError(f"No encontré subcarpetas de clases dentro de: {src_root}")

    ensure_dir(dst_root)

    total_in = 0
    total_out = 0

    for cls_dir in sorted(class_dirs):
        dst_cls = dst_root / cls_dir.name
        ensure_dir(dst_cls)

        imgs = [p for p in cls_dir.rglob("*") if p.is_file() and is_image_file(p)]
        if not imgs:
            print(f"[Aviso] Clase '{cls_dir.name}' no tiene imágenes.")
            continue

        for img_path in imgs:
            total_in += 1

            # Cargar imagen
            with Image.open(img_path) as im:
                im = im.convert("RGB")

                stem = img_path.stem

                # Copia del "original" (opcional) pero con resize/normalización de tamaño
                if copy_original:
                    im_base = transforms.Resize((imgsize, imgsize))(im)
                    out_name = f"{stem}__orig{out_ext}"
                    out_path = dst_cls / out_name
                    im_base.save(out_path, quality=jpg_quality)
                    total_out += 1

                # Augmentations
                for k in range(1, num_aug_per_image + 1):
                    # Nota: para tener aleatoriedad reproducible por imagen,
                    # puedes descomentar y usar un seed dependiente del path:
                    # random_seed = (hash(str(img_path)) + seed + k) & 0xFFFFFFFF
                    # torch.manual_seed(random_seed)

                    im_aug = aug(im)
                    out_name = f"{stem}__aug{k:02d}{out_ext}"
                    out_path = dst_cls / out_name
                    im_aug.save(out_path, quality=jpg_quality)
                    total_out += 1

        print(f"Clase '{cls_dir.name}': {len(imgs)} imgs -> salida en {dst_cls}")

    print("\nListo.")
    print(f"Imágenes entrada: {total_in}")
    print(f"Imágenes generadas (incluye orig si copy_original=True): {total_out}")
    print(f"Salida en: {dst_root}")


if __name__ == "__main__":
    # AJUSTA ESTO A TU RUTA REAL
    SRC_TEST = Path(r"C:\TAIA\challenge_grupo_2_2026\dataset_challenge\test")
    DST_TEST3 = Path(r"C:\TAIA\challenge_grupo_2_2026\dataset_challenge\test3")  

    augment_folder(
        src_root=SRC_TEST,
        dst_root=DST_TEST3,
        num_aug_per_image=3,   # genera 3 variantes por imagen (más el original si copy_original=True)
        imgsize=224,
        seed=42,
        copy_original=False,    # si no quieres copiar el original, pon False
        out_ext=".jpg",
        jpg_quality=95,
    )