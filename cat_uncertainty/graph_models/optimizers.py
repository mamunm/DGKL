"""Optimizers and learning rate schedulers for graph models."""

from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    ChainedScheduler,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
    StepLR,
    LinearLR,
    OneCycleLR,
)


def get_optimizer_and_scheduler(
    parameters: Any,
    optimizer_name: str,
    scheduler_name: str,
    learning_rate: float,
    max_epochs: int,
    total_steps: int,
) -> tuple[list[Optimizer], list[dict[str, Any]]]:
    """Configure optimizers and learning rate schedulers.

    Args:
        parameters: Model parameters to optimize
        optimizer_name: Name of the optimizer to use
        scheduler_name: Name of the scheduler to use
        learning_rate: Learning rate
        max_epochs: Maximum number of epochs
        total_steps: Total number of steps

    Returns:
        Tuple of optimizer and scheduler configuration
    """
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            parameters,
            lr=learning_rate,
            weight_decay=1e-5
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=1e-5
        )
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=0.9,
            weight_decay=1e-5
        )
    elif optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters,
            lr=learning_rate,
            momentum=0.9,
            weight_decay=1e-5
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # Set up scheduler
    if scheduler_name == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max_epochs,
        )
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
        }
    elif scheduler_name == "cosine_warm_restarts":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max_epochs//8,
            T_mult=2,
            eta_min=1e-6,
        )
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
        }
    elif scheduler_name == "reduce_on_plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        )
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss",
        }
    elif scheduler_name == "step":
        scheduler = StepLR(
            optimizer,
            step_size=30,
            gamma=0.1,
        )
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
        }
    elif scheduler_name == "fairchem":
        linear_scheduler = LinearLR(
            optimizer,
            start_factor=0.001,
            end_factor=1.0,
            total_iters=5,
        )
        step_scheduler = StepLR(
            optimizer,
            step_size=15,
            gamma=0.1,
        )
        scheduler = ChainedScheduler(
            [step_scheduler, linear_scheduler],
            optimizer=optimizer,
        )
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
        }
    elif scheduler_name == "onecycle":
        scheduler = OneCycleLR(
            optimizer, 
            max_lr=learning_rate, 
            total_steps=total_steps, 
            pct_start=0.3,
            anneal_strategy='cos', 
            cycle_momentum=True, 
            base_momentum=0.85, 
            max_momentum=0.95,
            div_factor=25., 
            final_div_factor=1e4
        )
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    return [optimizer], [scheduler_config]
