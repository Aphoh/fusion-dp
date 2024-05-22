import torch
from typing import Optional
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from . import lucas_processing
import pickle
from torch.utils.data import DataLoader
from torch import Generator
from pathlib import Path


def collate_fn(batch):
    inputs, labels, lengths = zip(*batch)
    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True).transpose(1, 2)
    labels = torch.tensor(labels)
    return inputs, labels, lengths


class DhruvaDataModule(pl.LightningDataModule):
    """
    DataModule for Lucas' fusion dataset.
    """

    # TODO: add sample rate here
    def __init__(
        self,
        data_dir: str,
        pin_memory: bool,
        file_path: str = "none",
        data_type: str = "default",
        end_cutoff_timesteps=8,
        machine_hyperparameters: dict = {"cmod": 1.0, "d3d": 1.0, "east": 1.0},
        batch_size: int = 32,
        test_batch_size: int = 32,
        num_workers: int = 1,
        augment: bool = False,
        debug: bool = False,
        seed: int = 42,
        len_aug_args: dict = {},
        taus: dict = {"cmod": 10, "d3d": 75, "east": 200},
        **kwargs,
    ):
        super().__init__()
        self.file_path = file_path
        self.pin_memory = pin_memory
        self.end_cutoff_timesteps = end_cutoff_timesteps
        self.machine_hyperparameters = machine_hyperparameters
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.augment = augment
        self.len_aug_args = len_aug_args
        self.debug = debug
        self.seed = seed
        self.taus = taus

        if data_type != "default" and data_type != "sequence":
            raise ValueError(f"data_type {data_type} not supported.")

        self.data_dim = 1
        self.data_type = "sequence"

        self.input_channels = 12
        self.output_channels = 1

        if end_cutoff_timesteps is None:
            raise ValueError("Must specify either end_cutoff_timesteps.")

    def setup(self, stage=None):
        # Load data from file
        assert Path(self.file_path).exists(), f"File {self.file_path} does not exist."
        with open(self.file_path, "rb") as f:
            shots = pickle.load(f)

        
        self.train_dataset = lucas_processing.ModelReadyDataset(
            shots=shots["train"].values(),
            inds=list(range(len(shots["train"]))),
            end_cutoff=None,
            end_cutoff_timesteps=self.end_cutoff_timesteps,
            machine_hyperparameters=self.machine_hyperparameters,
            taus=self.taus,
            len_aug=self.augment,
            len_aug_args=self.len_aug_args,
        )
        self.val_dataset = lucas_processing.ModelReadyDataset(
            shots=shots["val"].values(),
            inds=list(range(len(shots["val"]))),
            end_cutoff=None,
            end_cutoff_timesteps=self.end_cutoff_timesteps,
            machine_hyperparameters=self.machine_hyperparameters,
            taus=self.taus,
            len_aug=self.augment,
            len_aug_args=self.len_aug_args,
        )
        self.test_dataset = lucas_processing.ModelReadyDataset(
            shots=shots["test"].values(),
            inds=list(range(len(shots["test"]))),
            end_cutoff=None,
            end_cutoff_timesteps=self.end_cutoff_timesteps,
            machine_hyperparameters=self.machine_hyperparameters,
            taus=self.taus,
            len_aug=self.augment,
            len_aug_args=self.len_aug_args,
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        dl = DataLoader(
            self.train_dataset,
            self.batch_size,
            shuffle=True,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )
        return dl

    def test_dataloader(self) -> EVAL_DATALOADERS:
        dl = DataLoader(
            self.test_dataset,
            self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )
        return dl

    def val_dataloader(self) -> EVAL_DATALOADERS:
        dl = DataLoader(
            self.val_dataset,
            self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )
        return dl

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )
