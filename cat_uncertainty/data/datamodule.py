"""This module defines the PyTorch Lightning DataModule for graph data."""

from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from torch_geometric.data import Batch

from .config import DataModuleConfig
from .lmdb_dataset import LMDBDataset


def collate_fn(data_list):
    """Collate function to create a batch from a list of data objects."""
    return Batch.from_data_list(data_list)


class GraphDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for graph data.

    Args:
        config (DataModuleConfig): Configuration for the data module.
    """

    def __init__(self, config: DataModuleConfig):
        super().__init__()
        self.config = config
        self.collate_fn = collate_fn

    def construct_transform(self):
        """Constructs the transform function based on the configuration."""
        if self.config.transform is not None:
            from .soap_data import SOAPFeatureExtractor

            db_paths = get_db_paths(
                self.config.data_params.db_path,
                self.config.data_params.lmdb_files,
            )

            return SOAPFeatureExtractor(
                self.config.transform.params, db_paths=db_paths
            ).transform

    def setup(self, stage=None):
        """
        Setup datasets for training, validation, and testing.

        Args:
            stage (str, optional): Stage of setup process
                ('fit', 'test', or None).
        """
        transform = self.construct_transform()
        full_dataset = LMDBDataset(
            db_path=self.config.data_params.db_path,
            lmdb_files=self.config.data_params.lmdb_files,
            transform=transform,
        )
        total_size = len(full_dataset)
        train_size = int(
            total_size * self.config.data_params.train_fraction
        )
        val_size = int(total_size * self.config.data_params.val_fraction)
        test_size = total_size - train_size - val_size

        generator = torch.Generator().manual_seed(
            self.config.dataloader_params.seed
        )
        (
            train_dataset,
            val_dataset,
            test_dataset,
        ) = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=generator,
        )

        if stage == "fit" or stage is None:
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
        if stage == "test" or stage is None:
            self.test_dataset = test_dataset

    def state_dict(self):
        """
        Returns a dictionary containing the state of the data module.

        Args:
            None

        Returns:
            dict: Dictionary containing the state of the
                data module.
        """
        self.setup()
        return {
            "train_dataset": self.train_dataset.indices,
            "val_dataset": self.val_dataset.indices,
            "test_dataset": self.test_dataset.indices,
        }

    def load_state_dict(self, state_dict):
        """
        Loads the state of the data module from a dictionary.

        Args:
            state_dict (dict): Dictionary containing the state of the
                data module.
        """
        self.setup()
        full_dataset = LMDBDataset(
            db_path=self.config.data_params.db_path,
            lmdb_files=self.config.data_params.lmdb_files,
            transform=self.construct_transform(),
        )
        self.train_dataset = torch.utils.data.Subset(
            full_dataset, state_dict["train_dataset"]
        )
        self.val_dataset = torch.utils.data.Subset(
            full_dataset, state_dict["val_dataset"]
        )
        self.test_dataset = torch.utils.data.Subset(
            full_dataset, state_dict["test_dataset"]
        )

    def train_dataloader(self):
        """
        Returns the training data loader.

        Returns:
            DataLoader: Training data loader.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.dataloader_params.batch_size,
            num_workers=self.config.dataloader_params.num_workers,
            shuffle=self.config.dataloader_params.train_shuffle,
            collate_fn=self.collate_fn,
            pin_memory=self.config.dataloader_params.pin_memory,
            persistent_workers=self.config.dataloader_params.persistent_workers,
            prefetch_factor=self.config.dataloader_params.prefetch_factor,
        )

    def val_dataloader(self):
        """
        Returns the validation data loader.

        Returns:
            DataLoader: Validation data loader.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.dataloader_params.batch_size,
            num_workers=self.config.dataloader_params.num_workers,
            shuffle=self.config.dataloader_params.val_shuffle,
            collate_fn=self.collate_fn,
            pin_memory=self.config.dataloader_params.pin_memory,
            persistent_workers=self.config.dataloader_params.persistent_workers,
            prefetch_factor=self.config.dataloader_params.prefetch_factor,
        )

    def test_dataloader(self):
        """
        Returns the test data loader.

        Returns:
            DataLoader: Test data loader.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.dataloader_params.batch_size,
            num_workers=self.config.dataloader_params.num_workers,
            shuffle=self.config.dataloader_params.test_shuffle,
            collate_fn=self.collate_fn,
            pin_memory=self.config.dataloader_params.pin_memory,
            persistent_workers=self.config.dataloader_params.persistent_workers,
            prefetch_factor=self.config.dataloader_params.prefetch_factor,
        )

    def cal_dataloader(self):
        """
        Returns the calibration data loader.

        Returns:
            DataLoader: Calibration data loader.
        """
        return DataLoader(
            self.cal_dataset,
            batch_size=self.config.dataloader_params.batch_size,
            num_workers=self.config.dataloader_params.num_workers,
            shuffle=self.config.dataloader_params.cal_shuffle,
            collate_fn=self.collate_fn,
            pin_memory=self.config.dataloader_params.pin_memory,
            persistent_workers=self.config.dataloader_params.persistent_workers,
            prefetch_factor=self.config.dataloader_params.prefetch_factor,
        )
