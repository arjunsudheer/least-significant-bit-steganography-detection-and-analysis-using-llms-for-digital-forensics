import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class BinaryStegoDataset(Dataset):
    def __init__(self, samples, image_size: int = 512, augment: bool = False):
        self.samples = samples
        transforms = [T.CenterCrop(image_size)]
        if augment:
            transforms += [T.RandomHorizontalFlip(), T.RandomVerticalFlip()]
        transforms.append(T.ToTensor())
        self.transform = T.Compose(transforms)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return (
            self.transform(img),
            torch.tensor(label, dtype=torch.float32),
            os.path.basename(path),
        )


def _collect_from_splits(dataset_root: str, splits: list) -> list:
    samples = []
    for split in splits:
        split_dir = os.path.join(dataset_root, split)
        for class_name, label in {"clean": 0, "stego": 1}.items():
            class_dir = os.path.join(split_dir, class_name)
            if os.path.isdir(class_dir):
                for f in sorted(os.listdir(class_dir)):
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif")):
                        samples.append((os.path.join(class_dir, f), label))
    return samples


def get_train_val_samples(dataset_root: str) -> list:
    """Pool for K-Fold CV — excludes the held-out test split."""
    return _collect_from_splits(dataset_root, ["train", "val"])


def get_test_samples(dataset_root: str) -> list:
    """Held-out test split — never used during training or fold selection."""
    return _collect_from_splits(dataset_root, ["test"])


def get_all_samples(dataset_root: str) -> list:
    """All splits combined (only use when there is no separate held-out set)."""
    return _collect_from_splits(dataset_root, ["train", "val", "test"])
