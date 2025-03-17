"""Monte Carlo Dropout model for uncertainty estimation."""


import torch
import torch.nn as nn


class MCDropout(nn.Module):
    """Monte Carlo Dropout model for uncertainty estimation.
    Uses dropout at inference time to generate multiple predictions
    and estimate uncertainty.

    Args:
        model (nn.Module): Model to wrap
        num_forward_passes (int): Number of forward passes
    """

    def __init__(self, model: nn.Module, num_forward_passes: int = 10):
        super().__init__()
        self.model = model
        self.num_forward_passes = num_forward_passes

        def enable_dropout(m: nn.Module):
            if isinstance(m, nn.Dropout):
                m.train()

        self.model.apply(enable_dropout)

    def forward(
        self, z: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with Monte Carlo Dropout.

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
            [self.model(z, pos, batch) for _ in range(self.num_forward_passes)],
            dim=0,
        )

        mean = torch.mean(predictions, dim=0).view(-1)
        std = torch.std(predictions, dim=0).view(-1)

        return mean, std
