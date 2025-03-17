from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool

from .utils import get_activation


@dataclass
class GCNConvModelParams:
    """Parameters for GCNConv model architecture."""

    num_message_passes: int = field(
        default=3,
        metadata={"description": "Number of message passes"},
    )
    num_node_features: int = field(
        default=32,
        metadata={"description": "Number of node features"},
    )
    hidden_channels: int = field(
        default=32,
        metadata={"description": "Number of hidden channels"},
    )
    num_out_features: int = field(
        default=32,
        metadata={"description": "Number of output features"},
    )
    activation: Literal["relu", "gelu", "silu", "tanh", "ssp"] = field(
        default="relu",
        metadata={"description": "Activation function"},
    )
    dropout: float | None = field(
        default=None,
        metadata={"description": "Dropout rate"},
    )
    pool: Literal["mean", "add", "atomic"] = field(
        default="mean",
        metadata={"description": "Pooling type"},
    )

    def __post_init__(self) -> None:
        """Validate model parameters."""
        if self.num_message_passes <= 0:
            raise ValueError("Number of message passes must be positive")
        if self.num_node_features <= 0:
            raise ValueError("Number of node features must be positive")
        if self.hidden_channels <= 0:
            raise ValueError("Hidden channels must be positive")
        if self.num_out_features <= 0:
            raise ValueError("Number of output features must be positive")
        if self.dropout is not None and not 0 <= self.dropout <= 1:
            raise ValueError("Dropout rate must be between 0 and 1")
        if self.pool not in ["mean", "add", "atomic"]:
            raise ValueError("Pooling type must be one of: mean, add, atomic")


class GCNConvModel(torch.nn.Module):
    """
    GCNConv model architecture.

    Args:
        params (GCNConvModelParams): Model parameters
    """

    def __init__(self, params: GCNConvModelParams):
        super().__init__()
        self.pool = params.pool
        self.num_layers = params.num_message_passes
        self.conv_layers = nn.ModuleList(
            [
                GCNConv(params.num_node_features, params.hidden_channels),
                *[
                    GCNConv(params.hidden_channels, params.hidden_channels)
                    for _ in range(self.num_layers - 1)
                ],
            ]
        )

        self.batch_norms = nn.ModuleList(
            [
                nn.BatchNorm1d(params.hidden_channels)
                for _ in range(self.num_layers)
            ]
        )

        self.activation = get_activation(params.activation)
        self.dropout = (
            nn.Dropout(params.dropout)
            if params.dropout is not None
            else nn.Identity()
        )
        self.linear = nn.Linear(params.hidden_channels, params.num_out_features)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the GCNConv model.

        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Edge indices.
            batch (torch.Tensor): Batch indices.

        Returns:
            torch.Tensor: Output tensor.
        """
        for i in range(self.num_layers):
            x = self.conv_layers[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = self.activation(x)
            if i < self.num_layers - 1:
                x = self.dropout(x)

        if self.pool == "add":
            x = global_add_pool(x, batch)
        elif self.pool == "mean":
            x = global_mean_pool(x, batch)
        elif self.pool == "atomic":
            x = x
        else:
            raise ValueError("Pooling type must be one of: add, mean, atomic")
        out = self.linear(x)

        return out
