"""Lightning module for MCD models."""

import warnings
from dataclasses import dataclass, field
from typing import Literal

import pytorch_lightning as pl
import torch
import torch.nn as nn

from ..graph_models.optimizers import get_optimizer_and_scheduler
from ..graph_models.regression_metrics import RegressionMetrics

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pytorch_lightning.utilities.parsing",
)


@dataclass
class Optimizer:
    """Configuration for optimizer and learning rate scheduler."""

    optimizer: str = field(
        default="adam",
        metadata={"description": "Optimizer type"},
    )
    scheduler: str = field(
        default="cosine",
        metadata={"description": "Learning rate scheduler type"},
    )
    learning_rate: float = field(
        default=1e-3,
        metadata={"description": "Learning rate"},
    )

    def __post_init__(self) -> None:
        """Validate the optimizer configuration."""
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")


class MCDLightningModule(pl.LightningModule):
    """Lightning module for MCD models.

    Args:
        model (nn.Module): The UQ model to train
        optimizer (Optimizer): Training configuration
        max_epochs (int): Maximum number of epochs
        num_forward_passes (int): Number of forward passes
        loss_fn: "mse" | "nll"
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        max_epochs: int,
        num_forward_passes: int = 10,
        loss_fn: Literal["mse", "nll"] = "mse",
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.num_forward_passes = num_forward_passes
        self.loss_fn = loss_fn

        self.metrics = RegressionMetrics(
            metrics=["mae", "rmse", "r2"],
            stages=["train", "val", "test"],
        )
        self.save_hyperparameters()

    def forward(
        self, z: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of the UQ model.

        Args:
            z: Node features tensor [N, D] or atomic numbers [N]
            pos: Edge index tensor [2, E] or positions [N, 3]
            batch: Batch assignment tensor [N]

        Returns:
            torch.Tensor
        """
        return self.model(z, pos, batch)

    def criterion(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate ELBO loss for training.

        Args:
            z: Node features tensor [N, D] or atomic numbers [N]
            pos: Edge index tensor [2, E] or positions [N, 3]
                - batch: Batch assignment tensor [N]
            y: Target values [batch_size, output_dim]

        Returns:
            torch.Tensor: NIG loss + Regularization loss
        """

        def enable_dropout(m: nn.Module):
            if isinstance(m, nn.Dropout):
                m.train()

        self.model.apply(enable_dropout)

        predictions = torch.stack(
            [self.model(z, pos, batch) for _ in range(self.num_forward_passes)],
            dim=0,
        )

        mean = torch.mean(predictions, dim=0).view(-1)
        std = torch.std(predictions, dim=0).view(-1)

        nll_loss = (
            0.5 * torch.log(2 * torch.pi * std**2)
            + 0.5 * (y - mean) ** 2 / std**2
        )
        loss = nll_loss.mean()
        return loss

    def _shared_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, stage: str
    ) -> torch.Tensor:
        """Shared step for training, validation and testing."""

        z = batch["atomic_numbers"].long()
        pos = batch["pos"]
        graph_batch = batch["batch"]
        y_true = batch["energy"]

        y_pred = self.model(z, pos, graph_batch).view(-1)
        if self.loss_fn == "mse":
            loss = nn.MSELoss()(y_true, y_pred)
        else:
            loss = self.criterion(z, pos, graph_batch, y=y_true)

        self.metrics.update(y_pred, y_true, stage=stage)

        self.log(
            f"{stage}_loss",
            loss,
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
                    sync_dist=True,
                    batch_size=y_true.size(0),
                )

        return loss

    def on_train_epoch_end(self) -> None:
        """Called at the end of training epoch."""
        self.metrics.reset("train")

    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch."""
        self.metrics.reset("val")

    def on_test_epoch_end(self) -> None:
        """Called at the end of test epoch."""
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
            max_epochs=self.max_epochs,
        )
