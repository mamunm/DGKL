"""Lightning module for Evidential models."""

import warnings
from dataclasses import dataclass, field

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


class EvidentialLightningModule(pl.LightningModule):
    """Lightning module for Evidential models.

    Args:
        model (nn.Module): The UQ model to train
        optimizer (Optimizer): Training configuration
        max_epochs (int): Maximum number of epochs
        reg_coeff (float): Regularization coefficient
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        max_epochs: int,
        reg_coeff: float = 1.0,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.reg_coeff = reg_coeff

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
        mu, lambda_, alpha, beta = self(z, pos, batch)
        error = y - mu
        gamma = 2 * beta * (1 + lambda_)
        nig_loss = 0.5 * torch.log(torch.pi / lambda_) - alpha * torch.log(
            gamma
        )
        nig_loss += (alpha + 0.5) * torch.log(lambda_ * error**2 + gamma)
        nig_loss += torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
        reg_loss = (2 * lambda_ + alpha) * torch.abs(error)
        loss = nig_loss + self.reg_coeff * reg_loss
        return loss.mean()

    def _shared_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, stage: str
    ) -> torch.Tensor:
        """Shared step for training, validation and testing."""

        z = batch["atomic_numbers"].long()
        pos = batch["pos"]
        graph_batch = batch["batch"]
        y_true = batch["y_relaxed"]

        y_pred, _, _, _ = self.model(z, pos, graph_batch)
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
            max_epochs=self.optimizer.max_epochs,
            total_steps=self.optimizer.total_steps,
        )
