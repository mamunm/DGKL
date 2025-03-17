"""PaiNN (Polarizable Interaction Neural Network) Implementation.

This module implements the PaiNN architecture as described in the paper:
"Equivariant message passing for the prediction of tensorial properties
and molecular spectra" (Schütt et al., 2021). The model uses message
passing between atomic environments to predict molecular properties
while maintaining rotational and translational equivariance.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch_scatter import scatter

from .utils import radius_graph

N_CHUNK = 3


@dataclass
class PaiNNModelParams:
    """Parameters for PaiNN model architecture.

    This dataclass defines the parameters for the PaiNN model architecture.
    It includes the number of layers, hidden channels, number of radial basis
    functions, cutoff distance, maximum number of neighbors, number of atom
    types, activation function, output dimension, dropout rate, and readout
    type.

    Attributes:
        num_layers: Number of layers.
        hidden_channels: Number of hidden channels.
        num_radial: Number of radial basis functions.
        cutoff: Cutoff distance.
        max_neighbors: Maximum number of neighbors.
        n_atom_type: Number of atom types.
        activation: Activation function.
        output_dim: Output dimension.
        dropout: Dropout rate.
        readout: Readout type.
    """

    num_layers: int = field(
        default=3,
        metadata={"description": "Number of layers"},
    )
    hidden_channels: int = field(
        default=128,
        metadata={"description": "Number of hidden channels"},
    )
    num_radial: int = field(
        default=20,
        metadata={"description": "Number of radial basis functions"},
    )
    cutoff: float = field(
        default=5.0,
        metadata={"description": "Cutoff distance"},
    )
    max_neighbors: int = field(
        default=32,
        metadata={"description": "Maximum number of neighbors"},
    )
    n_atom_type: int = field(
        default=100,
        metadata={"description": "Number of atom types"},
    )
    activation: str = field(
        default="relu",
        metadata={"description": "Activation function"},
    )
    output_dim: int = field(
        default=1,
        metadata={"description": "Output dimension"},
    )
    dropout: float | None = field(
        default=None,
        metadata={"description": "Dropout rate"},
    )
    readout: str = field(
        default="mean",
        metadata={"description": "Pooling type"},
    )

    def __post_init__(self) -> None:
        """Validate model parameters."""
        if self.num_layers <= 0:
            raise ValueError("Number of layers must be positive")
        if self.hidden_channels <= 0:
            raise ValueError("Hidden channels must be positive")
        if self.num_radial <= 0:
            raise ValueError(
                "Number of radial basis functions must be positive"
            )
        if self.cutoff <= 0:
            raise ValueError("Cutoff must be positive")
        if self.max_neighbors <= 0:
            raise ValueError("Maximum number of neighbors must be positive")
        if self.n_atom_type <= 0:
            raise ValueError("Number of atom types must be positive")
        if self.activation not in ["relu", "gelu", "silu", "tanh", "ssp"]:
            raise ValueError(
                "Activation function must be one of: "
                "relu, gelu, silu, tanh, ssp"
            )
        if self.output_dim <= 0:
            raise ValueError("Output dimension must be positive")
        if self.dropout is not None and not 0 <= self.dropout <= 1:
            raise ValueError("Dropout rate must be between 0 and 1")
        if self.readout not in ["mean", "add", "atomic"]:
            raise ValueError("Readout must be one of: mean, add, atomic")


class RadialBasisLayer(nn.Module):
    """Radial basis function layer that transforms distances into feature basis.

    This layer converts interatomic distances into a set of radial basis
    functions, providing a continuous and differentiable representation of
    atomic environments.

    Args:
        num_radial: Number of radial basis functions to use
        cutoff: Cutoff radius for interactions
    """

    def __init__(self, num_radial: int = 20, cutoff: float = 5.0) -> None:
        super().__init__()
        self.num_radial = num_radial
        self.cutoff = cutoff

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """Transform distances into radial basis functions.

        Args:
            distances: Tensor of interatomic distances

        Returns:
            Tensor of radial basis function values
        """
        cos_cutoff = 0.5 * (torch.cos(torch.pi * distances / self.cutoff) + 1.0)
        cos_cutoff = torch.where(
            distances <= self.cutoff, cos_cutoff, torch.zeros_like(distances)
        )

        n = torch.arange(1, self.num_radial + 1, device=distances.device)
        rbf = torch.sin(
            n * torch.pi * distances.unsqueeze(-1) / self.cutoff
        ) / distances.unsqueeze(-1)
        return rbf * cos_cutoff.unsqueeze(-1)


class MessageBlock(nn.Module):
    """Message block for computing interactions between atomic environments.

    This block computes messages between atoms based on their scalar and vector
    features, implementing the message passing operation described in Fig 2(b)
    of the paper.

    Args:
        hidden_dim: Dimension of hidden features
        num_radial: Number of radial basis functions
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_radial: int = 20,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_radial = num_radial

        self.scalar_message_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim),
        )

        self.W = nn.Linear(num_radial, 3 * hidden_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize network parameters."""
        for layer in self.scalar_message_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.W.weight)
        self.W.bias.data.fill_(0)

    def forward(
        self,
        s_j: torch.Tensor,
        v_j: torch.Tensor,
        edge_index: torch.Tensor,
        edge_rbf: torch.Tensor,
        edge_vec: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute messages between atoms.

        Args:
            s_j: Scalar features of source nodes
            v_j: Vector features of source nodes
            edge_index: Edge indices connecting nodes
            edge_rbf: Radial basis features of edges
            edge_vec: Edge vectors

        Returns:
            Tuple of updated scalar and vector features
        """
        message_features = self.scalar_message_net(s_j)
        w_features = self.W(edge_rbf)

        src, dst = edge_index[0], edge_index[1]
        mf_j = message_features[src]
        v_j_j = v_j[src]

        w_s, w_vs, w_vv = torch.split(
            mf_j * w_features, self.hidden_dim, dim=-1
        )

        w_v = v_j_j * w_vs.unsqueeze(1) / np.sqrt(3) + w_vv.unsqueeze(
            1
        ) * edge_vec.unsqueeze(2)

        delta_s = torch.zeros_like(s_j)
        delta_v = torch.zeros_like(v_j)

        delta_s.index_add_(0, dst, w_s)
        delta_v.index_add_(0, dst, w_v)
        delta_v = delta_v / np.sqrt(self.hidden_dim)

        return delta_s, delta_v


class UpdateBlock(nn.Module):
    """Update block for refining atomic representations.

    This block updates both scalar and vector features of atoms based on their
    current representations, implementing the update operation in Fig 2(c) of
    the paper.

    Args:
        hidden_dim: Dimension of hidden features
    """

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        self.message_update_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 3),
        )
        self.vector_update_net = nn.Linear(
            hidden_dim, hidden_dim * 2, bias=False
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize network parameters."""
        for layer in self.message_update_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.vector_update_net.weight)

    def forward(
        self, s_i: torch.Tensor, v_i: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update atomic representations.

        Args:
            s_i: Scalar features of nodes
            v_i: Vector features of nodes

        Returns:
            Tuple of updated scalar and vector features
        """
        v_1, v_2 = torch.split(
            self.vector_update_net(v_i), self.hidden_dim, dim=-1
        )
        v_dot = torch.einsum("bij,bij->bj", v_1, v_2)
        v_dot = v_dot / np.sqrt(self.hidden_dim)

        v_norm = torch.sqrt(torch.sum(v_2**2, dim=1) + 1e-12)
        features = torch.cat([s_i, v_norm], dim=-1)

        update_features = self.message_update_net(features)
        a_ss, a_sv, a_v = torch.chunk(update_features, N_CHUNK, dim=-1)

        delta_s = v_dot * a_sv + a_ss
        delta_s = delta_s / np.sqrt(2)

        delta_v = v_1 * a_v.unsqueeze(-2)

        return delta_s, delta_v


class PaiNN(nn.Module):
    """Polarizable Interaction Neural Network (PaiNN) implementation.

    PaiNN is a message passing neural network that maintains rotational
    and translational equivariance by using both scalar and vector
    features to represent atomic environments.

    Args:
        config (PaiNNModelParams): Model configuration
    """

    def __init__(self, config: PaiNNModelParams) -> None:
        super().__init__()
        self.hidden_channels = config.hidden_channels
        self.cutoff = config.cutoff
        self.max_neighbors = config.max_neighbors
        self.n_atom_type = config.n_atom_type
        self.readout = config.readout

        self.embedding = nn.Embedding(self.n_atom_type, self.hidden_channels)
        self.rbf = RadialBasisLayer(config.num_radial, self.cutoff)

        self.message_blocks = nn.ModuleList(
            [
                MessageBlock(self.hidden_channels, config.num_radial)
                for _ in range(config.num_layers)
            ]
        )

        self.update_blocks = nn.ModuleList(
            [
                UpdateBlock(self.hidden_channels)
                for _ in range(config.num_layers)
            ]
        )

        self.output_network = nn.Sequential(
            nn.Linear(self.hidden_channels, self.hidden_channels // 2),
            nn.SiLU(),
            (
                nn.Dropout(config.dropout)
                if config.dropout is not None
                else nn.Identity()
            ),
            nn.Linear(self.hidden_channels // 2, config.output_dim),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize network parameters."""
        for layer in self.output_network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0)

    def forward(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass of PaiNN.

        Args:
            z: Atomic numbers
            pos: Atomic positions
            batch: Batch assignments for atoms

        Returns:
            Predicted molecular energy
        """
        s_i = self.embedding(z)
        v_i = torch.zeros(s_i.size(0), 3, s_i.size(1), device=pos.device)

        edge_index, edge_dist = radius_graph(
            pos, batch, self.cutoff, self.max_neighbors
        )
        edge_dist = torch.where(
            edge_dist < 0,
            torch.tensor(1e-6, device=edge_dist.device),
            edge_dist,
        )

        edge_vec = (
            pos[edge_index[0]] - pos[edge_index[1]]
        ) / edge_dist.unsqueeze(-1)
        edge_rbf = self.rbf(edge_dist)

        for message_block, update_block in zip(
            self.message_blocks, self.update_blocks, strict=False
        ):
            delta_sm, delta_vm = message_block(
                s_i, v_i, edge_index, edge_rbf, edge_vec
            )

            s_i = s_i + delta_sm
            v_i = v_i + delta_vm

            delta_s_u, delta_v_u = update_block(s_i, v_i)

            s_i = s_i + delta_s_u
            v_i = v_i + delta_v_u

        atomic_energies = self.output_network(s_i).squeeze(-1)

        if self.readout == "add":
            energy = scatter(atomic_energies, batch, dim=0, reduce="sum")
        elif self.readout == "mean":
            energy = scatter(atomic_energies, batch, dim=0, reduce="mean")
        elif self.readout == "atomic":
            energy = atomic_energies
        else:
            raise ValueError("Readout must be one of: add, mean, atomic")

        return energy
