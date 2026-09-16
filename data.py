"""Archive officielle PathMNIST et séparation sélection/calibration."""

import hashlib
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset


DATA_URL = "https://zenodo.org/records/10519652/files/pathmnist.npz?download=1"
DATA_MD5 = "a8b06965200029087d5bd730944a56c1"
CLASS_NAMES = [
    "Tissu adipeux", "Fond", "Débris", "Lymphocytes", "Mucus",
    "Muscle lisse", "Muqueuse colique normale", "Stroma associé au cancer",
    "Épithélium d'adénocarcinome colorectal",
]
CLASS_CODES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]


def download_data(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "pathmnist.npz"
    if not path.exists():
        temporary = path.with_suffix(".partial")
        urlretrieve(DATA_URL, temporary)
        with temporary.open("rb") as file:
            digest = hashlib.file_digest(file, "md5").hexdigest()
        if digest != DATA_MD5:
            raise ValueError("Somme MD5 de PathMNIST incorrecte ; téléchargement conservé en .partial")
        temporary.replace(path)
    with path.open("rb") as file:
        if hashlib.file_digest(file, "md5").hexdigest() != DATA_MD5:
            raise ValueError("L'archive locale diffère de l'archive officielle")
    return path


def load_split(archive, split):
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Partition inconnue : {split}")
    with np.load(archive, allow_pickle=False) as data:
        images = data[f"{split}_images"]
        labels = data[f"{split}_labels"].reshape(-1).astype(np.int64)
    return images, labels


def split_validation(labels, seed=2026):
    indices = np.arange(len(labels))
    return train_test_split(indices, test_size=0.5, stratify=labels, random_state=seed)


def smoke_indices(labels, size, seed=2026):
    indices, _ = train_test_split(
        np.arange(len(labels)), train_size=size, stratify=labels, random_state=seed
    )
    return indices


class ImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = torch.from_numpy(self.images[index].transpose(2, 0, 1).copy())
        return image.float().div(255), int(self.labels[index])


def make_loader(images, labels, batch_size=256, shuffle=False, seed=0):
    return DataLoader(
        ImageDataset(images, labels), batch_size=batch_size, shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed), num_workers=0,
    )


def flat_features(images):
    # Conversion unique en float64 : format de calcul du solveur L-BFGS.
    features = images.reshape(len(images), -1).astype(np.float64)
    features /= 255
    return features
