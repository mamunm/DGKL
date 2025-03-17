"""SchNet model implementation."""

from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn

from .utils import get_activation, radius_graph


@dataclass
class SchNetModelParams:
    """Parameters for SchNet model architecture."""

    hidden_channels: int = field(
        default=128,
        metadata={"description": "Number of hidden channels"},
    )
    num_filters: int = field(
        default=128,
        metadata={"description": "Number of filters"},
    )
    num_interactions: int = field(
        default=6,
        metadata={"description": "Number of interactions"},
    )
    cfconv_num_dense: int = field(
        default=2,
        metadata={"description": "Number of dense layers in CFConv"},
    )
    interaction_num_dense: int = field(
        default=2,
        metadata={"description": "Number of dense layers in interaction"},
    )
    num_gaussians: int = field(
        default=50,
        metadata={"description": "Number of Gaussians"},
    )
    cutoff: float = field(
        default=10.0,
        metadata={"description": "Cutoff distance"},
    )
    max_num_neighbors: int = field(
        default=32,
        metadata={"description": "Maximum number of neighbors"},
    )
    n_atom_type: int = field(
        default=100,
        metadata={"description": "Number of atom types"},
    )
    output_dim: int = field(
        default=1,
        metadata={"description": "Output dimension"},
    )
    activation: Literal["ssp"] = field(
        default="ssp",
        metadata={"description": "Activation function"},
    )
    readout: Literal["add", "mean", "atomic"] = field(
        default="mean",
        metadata={"description": "Readout function"},
    )
    dropout: float | None = field(
        default=None,
        metadata={"description": "Dropout rate"},
    )

    def __post_init__(self) -> None:
        """Validate model parameters."""
        if self.hidden_channels <= 0:
            raise ValueError("Hidden channels must be positive")
        if self.num_filters <= 0:
            raise ValueError("Number of filters must be positive")
        if self.num_interactions <= 0:
            raise ValueError("Number of interactions must be positive")
        if self.cfconv_num_dense <= 0:
            raise ValueError("CFConv dense layers must be positive")
        if self.interaction_num_dense <= 0:
            raise ValueError("Interaction dense layers must be positive")
        if self.num_gaussians <= 0:
            raise ValueError("Number of Gaussians must be positive")
        if self.cutoff <= 0:
            raise ValueError("Cutoff must be positive")
        if self.max_num_neighbors <= 0:
            raise ValueError("Maximum number of neighbors must be positive")
        if self.n_atom_type <= 0:
            raise ValueError("Number of atom types must be positive")
        if self.output_dim <= 0:
            raise ValueError("Output dimension must be positive")
        if self.readout not in ["add", "mean", "atomic"]:
            raise ValueError("Readout must be one of: add, mean, atomic")
        if self.dropout is not None and not 0 <= self.dropout <= 1:
            raise ValueError("Dropout rate must be between 0 and 1")


class GaussianSmearing(nn.Module):
    """
    Gaussian smearing of distances for continuous-filter convolutions.

    Args:
        start (float): Start value for gaussian centers.
        stop (float): Stop value for gaussian centers.
        num_gaussians (int): Number of gaussian basis functions.
    """

    def __init__(
        self, start: float = 0.0, stop: float = 10.0, num_gaussians: int = 50
    ):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """
        Expands distances in gaussian basis functions.

        Args:
            dist (torch.Tensor): Distance values [N].

        Returns:
            torch.Tensor: Expanded distances [N, num_gaussians].
        """
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ContinuousFilterConv(nn.Module):
    """
    Continuous-filter convolution layer for SchNet.

    This layer implements the continuous-filter convolution operation used in
    SchNet. It transforms node features using learned filters that are
    continuous functions of interatomic distances.

    Args:
        hidden_channels: Hidden channel dimension
        num_gaussians: Number of gaussians for distance expansion
        num_filters: Number of filters in MLPs
        activation: Activation function to use
        num_dense: Number of dense layers in continuous filter convolution
    """

    def __init__(
        self,
        hidden_channels: int,
        num_gaussians: int,
        num_filters: int,
        activation: str,
        num_dense: int = 2,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_gaussians = num_gaussians
        self.num_filters = num_filters

        self.lin1 = nn.Linear(hidden_channels, num_filters)
        self.lin2 = nn.Linear(num_filters, hidden_channels)

        mlp_layers = []
        in_channels = num_gaussians

        for _ in range(num_dense):
            mlp_layers.append(nn.Linear(in_channels, num_filters))
            mlp_layers.append(get_activation(activation))
            in_channels = num_filters

        self.mlp = nn.Sequential(*mlp_layers)

        self.reset_parameters()

    def reset_parameters(self):
        """Resets all learnable parameters of the module."""
        nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of continuous-filter convolution.

        Args:
            x (torch.Tensor): Node features [N, hidden_channels].
            edge_index (torch.Tensor): Edge indices [2, E].
            edge_attr (torch.Tensor): Edge features [E, num_gaussians].

        Returns:
            torch.Tensor: Updated node features [N, hidden_channels].
        """
        row, col = edge_index
        x_i = self.lin1(x)

        # Transform edge features
        edge_weight = self.mlp(edge_attr)

        # Message passing
        x_j = x_i[col]  # [E, num_filters]
        x_j = x_j * edge_weight  # [E, num_filters]

        # Aggregate messages
        out = torch.zeros_like(x_i)
        out.index_add_(0, row, x_j)

        # Update node embeddings
        out = self.lin2(out)

        return out


class SchNetInteraction(nn.Module):
    """
    Interaction block for SchNet, combining continuous-filter convolution
    with residual connection.

    Args:
        hidden_channels (int): Dimension of hidden features.
        num_gaussians (int): Number of gaussians for distance expansion.
        num_filters (int): Number of filters in MLPs.
        activation (str): Activation function to use.
        cfconv_num_dense (int): Number of dense layers in continuous filter
            convolution.
        interaction_num_dense (int): Number of dense layers in interaction
            update network.
    """

    def __init__(
        self,
        hidden_channels: int,
        num_gaussians: int,
        num_filters: int,
        activation: str,
        cfconv_num_dense: int,
        interaction_num_dense: int,
    ):
        super().__init__()
        self.conv = ContinuousFilterConv(
            hidden_channels,
            num_gaussians,
            num_filters,
            activation,
            cfconv_num_dense,
        )

        update_layers = []
        in_channels = hidden_channels

        for _ in range(interaction_num_dense):
            update_layers.extend(
                [
                    nn.Linear(in_channels, hidden_channels),
                    get_activation(activation),
                ]
            )
            in_channels = hidden_channels

        update_layers.append(nn.Linear(hidden_channels, hidden_channels))

        self.update_net = nn.Sequential(*update_layers)

        self.reset_parameters()

    def reset_parameters(self):
        """Resets all learnable parameters of the module."""
        self.conv.reset_parameters()
        for layer in self.update_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of interaction block.

        Args:
            x (torch.Tensor): Node features [N, hidden_channels].
            edge_index (torch.Tensor): Edge indices [2, E].
            edge_attr (torch.Tensor): Edge features [E, num_gaussians].

        Returns:
            torch.Tensor: Updated node features [N, hidden_channels].
        """
        v = self.conv(x, edge_index, edge_attr)
        v = self.update_net(v)
        return x + v


class SchNet(nn.Module):
    """
    SchNet model for learning molecular properties.

    Args:
        config (SchNetConfig): Model configuration parameters.
    """

    def __init__(self, config: SchNetModelParams):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(
            config.n_atom_type, config.hidden_channels
        )
        self.distance_expansion = GaussianSmearing(
            0.0, config.cutoff, config.num_gaussians
        )
        self.interactions = nn.ModuleList(
            [
                SchNetInteraction(
                    config.hidden_channels,
                    config.num_gaussians,
                    config.num_filters,
                    config.activation,
                    config.cfconv_num_dense,
                    config.interaction_num_dense,
                )
                for _ in range(config.num_interactions)
            ]
        )

        output_layers = [
            nn.Linear(config.hidden_channels, config.hidden_channels // 2),
            get_activation(config.activation),
        ]

        if config.dropout is not None and config.dropout > 0:
            output_layers.append(nn.Dropout(p=config.dropout))

        output_layers.append(
            nn.Linear(config.hidden_channels // 2, config.output_dim)
        )

        self.output_net = nn.Sequential(*output_layers)

        self.reset_parameters()

    def reset_parameters(self):
        """Resets all learnable parameters of the model."""
        nn.init.xavier_uniform_(self.embedding.weight)
        for interaction in self.interactions:
            interaction.reset_parameters()
        for layer in self.output_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0)

    def forward(
        self, z: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of SchNet.

        Args:
            z (torch.Tensor): Atomic numbers [N].
            pos (torch.Tensor): Atomic positions [N, 3].
            batch (torch.Tensor): Batch assignments [N].

        Returns:
            torch.Tensor: Predicted properties [batch_size, output_dim]
        """
        x = self.embedding(z)

        edge_index, edge_length = radius_graph(pos, batch, self.config.cutoff)
        edge_attr = self.distance_expansion(edge_length)

        for interaction in self.interactions:
            x = x + interaction(x, edge_index, edge_attr)

        x = self.output_net(x)

        if self.config.readout == "add":
            out = torch.zeros(
                (batch.max().item() + 1, x.size(1)),
                device=x.device,
                dtype=x.dtype,
            )
            out.index_add_(0, batch, x)
        elif self.config.readout == "mean":
            out = torch.zeros(
                (batch.max().item() + 1, x.size(1)),
                device=x.device,
                dtype=x.dtype,
            )
            count = torch.zeros(
                batch.max().item() + 1, device=x.device, dtype=torch.long
            )
            out.index_add_(0, batch, x)
            count.index_add_(0, batch, torch.ones_like(batch))
            out = out / count.unsqueeze(1).clamp(min=1)
        elif self.config.readout == "atomic":
            out = x
        else:
            raise ValueError("Readout type must be one of: add, mean, atomic")

        return out
