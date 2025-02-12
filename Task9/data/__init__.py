from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl
import torch
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

RAW_DATA_DIR = Path(__file__).parent / 'raw_data'

trasform = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),
])


class TripletDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: CIFAR10):
        self.dataset = dataset

    def _get_indices(self, idx, gen: torch.Generator | None):
        anchor_label = self.dataset[idx][1]

        positive_mask = torch.eq(torch.tensor(self.dataset.targets), torch.tensor(anchor_label)).ravel()
        positive_indices = torch.nonzero(positive_mask).ravel()
        positive_indices = positive_indices[positive_indices != idx]  # remove anchor idx from possible positives
        negative_indices = torch.nonzero(~positive_mask).ravel()

        positive_idx = positive_indices[torch.randint(len(positive_indices), (1,), generator=gen)].item()
        negative_idx = negative_indices[torch.randint(len(negative_indices), (1,), generator=gen)].item()

        return positive_idx, negative_idx

    def __len__(self):
        return len(self.dataset.targets)

    def __getitem__(self, index):
        gen = torch.Generator()
        gen.manual_seed(42)
        positive_idx, negative_idx = self._get_indices(index, gen)
        return self.dataset[index][0], self.dataset[positive_idx][0], self.dataset[negative_idx][0]


class CifarTripletDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.args = {
            'batch_size': 32,
        }
        self.train_dataset = None

    def prepare_data(self):
        self.train_dataset = CIFAR10(root=RAW_DATA_DIR, train=True, download=True, transform=trasform)

    def setup(self, stage=None):
        self.train_dataset = TripletDataset(CIFAR10(root=RAW_DATA_DIR, train=True, download=False, transform=trasform))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.args, shuffle=True)


class CifarDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.args = {
            'batch_size': 32,
        }
        self.train_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        self.train_dataset = CIFAR10(root=RAW_DATA_DIR, train=True, download=True, transform=trasform)
        self.test_dataset = CIFAR10(root=RAW_DATA_DIR, train=False, download=True, transform=trasform)

    def setup(self, stage=None):
        self.train_dataset = CIFAR10(root=RAW_DATA_DIR, train=True, download=False, transform=trasform)
        self.test_dataset = CIFAR10(root=RAW_DATA_DIR, train=False, download=False, transform=trasform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.args, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.args, shuffle=False)
