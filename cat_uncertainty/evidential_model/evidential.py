import torch
import torch.nn as nn
from torch.nn.functional import softplus


class EvidentialModel(nn.Module):
    """Evidential Regression model for uncertainty estimation.

    Wraps a base model and modifies its output to predict parameters
    of a Normal-Inverse-Gamma distribution (μ, λ, α, β) for each target
    dimension.

    Args:
        graph_model (nn.Module): Base model to wrap with evidential regression
        min_evidence (float): Minimum evidence value for numerical stability
    """

    def __init__(self, graph_model: nn.Module, min_evidence: float = 1e-5):
        super().__init__()
        self.graph_model = graph_model
        self.min_evidence = min_evidence

    def _positive_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Transform network outputs to positive values using softplus."""
        return softplus(x) + self.min_evidence

    def forward(
        self, z: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Forward pass with evidential regression.

        Args:
            z (torch.Tensor): Atomic numbers [N]
            pos (torch.Tensor): Atomic positions [N, 3]
            batch (torch.Tensor): Batch assignments [N]

        Returns:
            Tuple containing:
                - mean (μ) predictions [batch_size, output_dim]
                - total variance [batch_size, output_dim]
                - lambda (λ) precision parameter [batch_size, output_dim]
                - alpha (α) shape parameter [batch_size, output_dim]
                - beta (β) inverse scale parameter [batch_size, output_dim]
        """
        outputs = self.graph_model(z, pos, batch)

        mu = outputs[..., 0]
        lambda_ = self._positive_transform(outputs[..., 1])
        alpha = self._positive_transform(outputs[..., 2]) + 1.0
        beta = self._positive_transform(outputs[..., 3])

        return mu, lambda_, alpha, beta
