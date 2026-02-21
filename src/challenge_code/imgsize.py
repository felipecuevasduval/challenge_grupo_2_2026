from pathlib import Path
from PIL import Image
from collections import Counter

ROOT = Path(r"C:\TAIA\challenge_grupo_2_2026\dataset_challenge\train")

sizes = []
bad = 0

# toma extensiones típicas
exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for p in ROOT.rglob("*"):
    if p.suffix.lower() in exts:
        try:
            with Image.open(p) as im:
                sizes.append(im.size)  # (W, H)
        except Exception:
            bad += 1

c = Counter(sizes)

print(f"Total imágenes leídas: {len(sizes)}")
print(f"Imágenes con error: {bad}")
print(f"Tamaños distintos: {len(c)}")
print("\nTop 10 tamaños más comunes:")
for (w, h), n in c.most_common(10):
    print(f"  {w}x{h}: {n}")

if sizes:
    ws = [s[0] for s in sizes]
    hs = [s[1] for s in sizes]
    print("\nResumen:")
    print(f"  W min/max: {min(ws)} / {max(ws)}")
    print(f"  H min/max: {min(hs)} / {max(hs)}")