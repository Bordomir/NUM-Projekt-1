import os

import kagglehub
import pytorch_lightning as pl
import torch
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader, random_split

from dataset import CustomImageDataset


class CatVSDogDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = v2.Compose(
            [
                v2.Resize((128, 128)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize((0.5,), (0.5,)),
            ]
        )

    def prepare_data(self):
        kagglehub.dataset_download(
            "karakaggle/kaggle-cat-vs-dog-dataset", output_dir=self.data_dir
        )

        print("Path to dataset files:", self.data_dir)

        return super().prepare_data()

    def setup(self, stage=None):
        full_data_dir = os.path.join(
            self.data_dir, "kagglecatsanddogs_3367a", "PetImages"
        )

        full_dataset = CustomImageDataset(full_data_dir, self.transform)

        self.train, self.val, self.test = random_split(
            full_dataset, [0.6, 0.2, 0.2], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test, batch_size=self.batch_size, num_workers=self.num_workers
        )
