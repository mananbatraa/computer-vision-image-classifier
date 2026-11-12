"""Dataset download, stratified splits and image preprocessing."""
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# ImageNet statistics: the pretrained backbone expects inputs normalised this way.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transform(size: int = 128):
    """Augmentation suited to overhead imagery: any rotation or flip is still a valid image."""
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply([transforms.RandomRotation(90)], p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform(size: int = 128):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class Transformed(Dataset):
    """Apply a split-specific transform on top of an untransformed subset."""

    def __init__(self, subset, transform):
        self.subset, self.transform = subset, transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, i):
        image, label = self.subset[i]
        return self.transform(image.convert("RGB")), label


def load_eurosat(download: bool = True):
    from torchvision.datasets import EuroSAT

    return EuroSAT(str(DATA_DIR), download=download)


def stratified_splits(labels, val=0.15, test=0.15, seed=42, limit_per_class=None):
    """Return train/val/test index arrays with the same class balance in each."""
    labels = np.asarray(labels)
    idx = np.arange(len(labels))
    if limit_per_class:
        rng = np.random.default_rng(seed)
        idx = np.concatenate([rng.permutation(np.flatnonzero(labels == c))[:limit_per_class]
                              for c in np.unique(labels)])
    train_idx, rest = train_test_split(idx, test_size=val + test, stratify=labels[idx], random_state=seed)
    val_idx, test_idx = train_test_split(rest, test_size=test / (val + test), stratify=labels[rest],
                                         random_state=seed)
    return train_idx, val_idx, test_idx


def make_loaders(dataset, splits, size=128, batch_size=64, workers=0):
    train_idx, val_idx, test_idx = splits
    make = lambda idx, tf, shuffle: DataLoader(  # noqa: E731
        Transformed(Subset(dataset, idx), tf), batch_size=batch_size, shuffle=shuffle,
        num_workers=workers, persistent_workers=workers > 0)
    return (make(train_idx, train_transform(size), True),
            make(val_idx, eval_transform(size), False),
            make(test_idx, eval_transform(size), False))


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
