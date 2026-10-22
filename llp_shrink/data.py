"""Fixed, disjoint random bags; raw train labels used only to form bag counts.

The three official train/test splits are kept separate. No SVHN extra split.
Workers restart and are reseeded every epoch so epoch-boundary resume is exact.
"""
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms as T
from .vendor.rand_rgb import RandomAugment as RGBRandAugment
from .vendor.rand_gray import RandomAugment as GrayRandAugment


SPECS = {
    "CIFAR10": (3, 32, (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
    "SVHN": (3, 32, (0.4380, 0.4440, 0.4730), (0.1751, 0.1771, 0.1744)),
    # Normalization used for the grayscale experiment configuration.
    "FashionMNIST": (1, 28, (0.1307,), (0.3081,)),
}


def raw_dataset(name, root, train, download=False):
    if name == "CIFAR10":
        ds = datasets.CIFAR10(root, train=train, download=download)
        images, labels = ds.data, ds.targets
    elif name == "FashionMNIST":
        ds = datasets.FashionMNIST(root, train=train, download=download)
        images, labels = ds.data.numpy(), ds.targets.numpy()
    elif name == "SVHN":
        ds = datasets.SVHN(root, split="train" if train else "test", download=download)
        images, labels = ds.data.transpose(0, 2, 3, 1), ds.labels
    else:
        raise ValueError(f"unsupported dataset: {name}")
    return np.ascontiguousarray(images), np.asarray(labels, dtype=np.int64)


def synthetic_dataset(name, n, seed):
    """Explicitly synthetic integration fixtures, never claimed as real data."""
    channels, size, _, _ = SPECS[name]
    rng = np.random.default_rng(seed)
    shape = (n, size, size) if channels == 1 else (n, size, size, channels)
    return rng.integers(0, 256, shape, dtype=np.uint8), np.arange(n, dtype=np.int64) % 10


def fingerprint(images, labels):
    h = hashlib.sha256()
    for array in (images, labels):
        a = np.ascontiguousarray(array)
        h.update(str((a.shape, str(a.dtype))).encode())
        h.update(memoryview(a).cast("B"))
    return h.hexdigest()


@dataclass
class BagPartition:
    indices: np.ndarray
    counts: np.ndarray
    omitted: np.ndarray

    @classmethod
    def create(cls, labels, bag_size, seed, classes=10):
        if bag_size < 1 or len(labels) < bag_size:
            raise ValueError("dataset must contain at least one full bag")
        labels = np.asarray(labels)
        if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("labels must be a one-dimensional integer array")
        if np.any((labels < 0) | (labels >= classes)):
            raise ValueError("labels outside valid class range")
        # Use a dedicated seeded RNG for reproducible bag membership.
        order = np.random.RandomState(seed).permutation(len(labels))
        usable = len(labels) // bag_size * bag_size
        indices = order[:usable].reshape(-1, bag_size)
        counts = np.stack([np.bincount(labels[row], minlength=classes) for row in indices])
        return cls(indices, counts, order[usable:])

    def save(self, path):
        np.savez_compressed(path, indices=self.indices, counts=self.counts, omitted=self.omitted)


class AugmentPair:
    def __init__(self, name):
        channels, size, mean, std = SPECS[name]
        crop = [T.RandomCrop(size, padding=4, padding_mode="reflect")]
        if name != "SVHN":
            crop.append(T.RandomHorizontalFlip())
        self.crop = T.Compose(crop)
        self.normalize = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        self.rand = GrayRandAugment(2, 10) if channels == 1 else RGBRandAugment(2, 10)

    def __call__(self, image):
        image = Image.fromarray(image)
        weak = self.normalize(self.crop(image))
        strong_array = self.rand(np.asarray(self.crop(image)).copy())
        if strong_array.ndim == 3 and strong_array.shape[-1] == 1:
            strong_array = strong_array[..., 0]
        strong = self.normalize(Image.fromarray(np.ascontiguousarray(strong_array)))
        return weak, strong


class BagDataset(Dataset):
    def __init__(self, images, partition, name, audit_labels=None):
        self.images, self.partition = images, partition
        self.transform = AugmentPair(name)
        self.audit_labels = audit_labels

    def __len__(self):
        return len(self.partition.indices)

    def __getitem__(self, index):
        ids = self.partition.indices[index]
        pairs = [self.transform(self.images[i]) for i in ids]
        result = {
            "weak": torch.stack([x[0] for x in pairs]),
            "strong": torch.stack([x[1] for x in pairs]),
            "counts": torch.from_numpy(self.partition.counts[index].copy()),
            "ids": torch.from_numpy(ids.copy()), "bag_id": index,
        }
        if self.audit_labels is not None:
            result["audit_labels"] = torch.from_numpy(self.audit_labels[ids].copy())
        return result


class EvalDataset(Dataset):
    def __init__(self, images, labels, name):
        self.images, self.labels = images, labels
        _, _, mean, std = SPECS[name]
        self.transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.transform(self.images[index]), int(self.labels[index])


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)
    # OpenCV threads inside workers cause oversubscription on rental servers.
    import cv2
    cv2.setNumThreads(0)


def train_loader(dataset, batch_bags, workers, seed, epoch, pin_memory=False):
    if len(dataset) < batch_bags:
        raise ValueError("not enough bags for one complete batch; reduce batch_instances")
    generator = torch.Generator().manual_seed(seed + epoch * 100003)
    return DataLoader(dataset, batch_size=batch_bags, shuffle=True, drop_last=True,
                      num_workers=workers, pin_memory=pin_memory,
                      worker_init_fn=seed_worker, generator=generator,
                      persistent_workers=False)


def eval_loader(dataset, batch_size, workers, pin_memory=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                      num_workers=workers, pin_memory=pin_memory,
                      worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(0))
