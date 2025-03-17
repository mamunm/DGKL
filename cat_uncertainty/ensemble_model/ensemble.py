"""Ensemble model for uncertainty estimation."""


import torch
import torch.nn as nn


class EnsembleModel(nn.Module):
    """Ensemble model that combines multiple models for uncertainty estimation.
    All models share the same architecture but are trained on different subsets
    of data.

    Args:
        models (List[nn.Module]): List of models to combine
    """

    def __init__(self, models: list[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(
        self, z: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of ensemble model.

        Args:
            z (torch.Tensor): Atomic numbers [N]
            pos (torch.Tensor): Atomic positions [N, 3]
            batch (torch.Tensor): Batch assignments [N]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing:
                - mean predictions [batch_size, output_dim]
                - standard deviation of predictions [batch_size, output_dim]
        """
        predictions = torch.stack(
            [model(z, pos, batch) for model in self.models], dim=0
        )
        mean = torch.mean(predictions, dim=0)
        std = torch.std(predictions, dim=0)

        return mean, std
