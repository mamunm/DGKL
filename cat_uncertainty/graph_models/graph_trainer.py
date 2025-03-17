"""Graph Neural Network trainer module."""

import warnings
from dataclasses import dataclass, field
from typing import Literal

import pytorch_lightning as pl
import torch
import torch.nn as nn

from .optimizers import get_optimizer_and_scheduler
from .regression_metrics import RegressionMetrics

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pytorch_lightning.utilities.parsing",
)


@dataclass
class Optimizer:
    """Configuration for optimizer and learning rate scheduler."""

    optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = field(
        default="adam",
        metadata={"description": "Optimizer type"},
    )
    scheduler: Literal[
        "cosine",
        "cosine_warm_restarts",
        "reduce_on_plateau",
        "step",
        "fairchem",
        "onecycle",
    ] = field(
        default="onecycle",
        metadata={"description": "Learning rate scheduler type"},
    )
    learning_rate: float = field(
        default=1e-3,
        metadata={"description": "Learning rate"},
    )

    max_epochs: int = field(
        default=100,
        metadata={"description": "Maximum number of epochs"},
    )

    total_steps: int = field(
        default=100,
        metadata={"description": "Total number of steps"},
    )

    def __post_init__(self) -> None:
        """Validate the optimizer configuration."""
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")


class GraphLightningModule(pl.LightningModule):
    """Lightning module for graph neural networks.

    Args:
        model (nn.Module): The neural network model to train
        optimizer (Optimizer): The optimizer config
        max_epochs (int): Maximum number of epochs
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        max_epochs: int,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        # self.criterion = nn.MSELoss()
        self.criterion = nn.HuberLoss()

        self.metrics = RegressionMetrics(
            metrics=["mae", "rmse", "r2"],
            stages=["train", "val", "test"],
        )
        self.save_hyperparameters()

    def forward(
        self, z: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        return self.model(z, pos, batch)

    def _shared_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, stage: str
    ) -> torch.Tensor:
        """Shared step for training, validation and testing."""
        z = batch["atomic_numbers"].long()
        pos = batch["pos"]
        graph_batch = batch["batch"]
        y_true = batch["y_relaxed"]

        y_pred = self.model(z, pos, graph_batch)
        if y_pred.dim() != y_true.dim():
            y_pred = y_pred.view(y_true.shape)

        loss = self.criterion(y_pred, y_true)

        self.metrics.update(y_pred, y_true, stage=stage)
        self.log(
            f"{stage}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=y_true.size(0),
        )

        if stage == "train":
            self.log(
                "lr",
                self.optimizers().param_groups[0]["lr"],
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                batch_size=y_true.size(0),
            )

        if stage != "train":
            metrics = self.metrics.compute(stage)
            for name, value in metrics.items():
                self.log(
                    f"{stage}_{name}",
                    value,
                    prog_bar=True,
                    batch_size=y_true.size(0),
                )

        return loss

    def on_train_epoch_end(self) -> None:
        self.metrics.reset("train")

    def on_validation_epoch_end(self) -> None:
        self.metrics.reset("val")

    def on_test_epoch_end(self) -> None:
        self.metrics.reset("test")

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        metrics = self._shared_step(batch, batch_idx, "train")
        return metrics

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> None:
        self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        self._shared_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        return get_optimizer_and_scheduler(
            parameters=self.parameters(),
            optimizer_name=self.optimizer.optimizer,
            scheduler_name=self.optimizer.scheduler,
            learning_rate=self.optimizer.learning_rate,
            max_epochs=self.optimizer.max_epochs,
            total_steps=self.optimizer.total_steps,
        )
