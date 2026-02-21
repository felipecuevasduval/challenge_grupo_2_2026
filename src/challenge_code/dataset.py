from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import datasets


class ChallengeImageFolderDataset(Dataset):
    """
    Dataset basado en estructura de carpetas:
      root/
        A/
        B/
        ...
        (carpetas = clases)

    Usa torchvision.datasets.ImageFolder internamente.
    """

    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.data = datasets.ImageFolder(root=str(self.root), transform=transform)

    @property
    def classes(self):
        return self.data.classes

    @property
    def class_to_idx(self):
        return self.data.class_to_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        return image, label

    def sanity_check(self, n=5):
        """
        Imprime algunos ejemplos (shape y label) para verificar que todo esté ok.
        """
        n = min(n, len(self.data))
        for i in range(n):
            img, lab = self.data[i]
            if isinstance(img, torch.Tensor):
                print(i, "img:", tuple(img.shape), "label:", lab, "class:", self.data.classes[lab])
            else:
                print(i, "img:", img.size, "label:", lab, "class:", self.data.classes[lab])