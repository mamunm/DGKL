"""DimeNet++ implementation.

This module implements the DimeNet++ model, an extension of the Directional
Message Passing Neural Network (DimeNet) for molecular property prediction.

References:
    [1] Klicpera, J., Groß, J., & Günnemann, S. (2020).
        Directional message passing for molecular graphs.
        ICLR 2020.
    [2] Klicpera, J., Giri, S., Margraf, J. T., & Günnemann, S. (2020).
        Fast and Uncertainty-Aware Directional Message Passing for
        Non-Equilibrium Molecules. arXiv:2011.14115.
"""

from collections.abc import Callable

import numpy as np
import sympy as sym  # type: ignore
import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor
from torch_geometric.nn.inits import glorot_orthogonal  # type: ignore
from torch_geometric.nn.models.dimenet_utils import (  # type: ignore
    bessel_basis,
    real_sph_harm,
)
from torch_scatter import scatter  # type: ignore

from .utils import Activation, find_triplets, get_activation, radius_graph


class DimeNetConfig(BaseModel):
    """Configuration for DimeNet model.

    Network Architecture Args:
        hidden_channels (int): Hidden embedding dimension
        out_channels (int): Output embedding dimension
        int_emb_size (int): Embedding size for interaction triplets
        basis_emb_size (int): Embedding size for basis transformation
        num_blocks (int): Number of interaction blocks
        num_spherical (int): Number of spherical harmonics
        num_radial (int): Number of radial basis functions
        out_emb_size (int): Output block embedding size
        num_before_skip (int): Number of residual layer before skip connection
        num_after_skip (int): Number of residual layer after skip connection
        num_output_layers (int): Number of output layers

    Geometric Args:
        cutoff (float): Cutoff distance for interactions
        max_num_neighbors (int): Maximum number of neighbors
        envelope_exponent (int): Exponent in envelope function

    Input/Output Args:
        n_atom_type (int): Number of atom types

    Other Args:
        dropout (Optional[float]): Dropout rate
        activation (str): Activation function
    """

    # Network architecture
    hidden_channels: int = Field(128, description="Hidden embedding dimension")
    out_channels: int = Field(128, description="Output embedding dimension")
    int_emb_size: int = Field(64, description="Interaction embedding size")
    basis_emb_size: int = Field(8, description="Basis embedding size")
    num_blocks: int = Field(6, description="Number of building blocks")
    num_spherical: int = Field(7, description="Number of spherical harmonics")
    num_radial: int = Field(6, description="Number of radial basis functions")
    out_emb_size: int = Field(256, description="Output embedding size")
    num_before_skip: int = Field(1, description="Layers before skip connection")
    num_after_skip: int = Field(2, description="Layers after skip connection")
    num_output_layers: int = Field(3, description="Number of output layers")

    # Geometric parameters
    cutoff: float = Field(5.0, description="Cutoff radius")
    max_num_neighbors: int = Field(
        32, description="Maximum number of neighbors"
    )
    envelope_exponent: int = Field(5, description="Envelope exponent")

    # Input/Output parameters
    n_atom_type: int = Field(100, description="Number of atom types")

    # Other parameters
    dropout: float | None = None
    activation: Activation = Field(
        Activation.RELU, description="Activation function"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Envelope(nn.Module):
    """Envelope function that ensures a smooth cutoff."""

    def __init__(self, exponent: int):
        super().__init__()
        self.p = exponent
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p, a, b, c = self.p, self.a, self.b, self.c
        x_pow_p0 = x.pow(p)
        x_pow_p1 = x_pow_p0 * x
        x_pow_p2 = x_pow_p1 * x
        return (1.0 / x + a * x_pow_p0 + b * x_pow_p1 + c * x_pow_p2) * (
            x < 1.0
        ).to(x.dtype)


class EmbeddingBlock(nn.Module):
    """
    Initial embedding block for atoms.

    Args:
        n_atom_type (int): Number of atom types
        activation (str): Activation function
        emb_size (int, optional): Embedding size. Defaults to 128.
    """

    def __init__(
        self,
        n_atom_type,
        num_radial,
        hidden_channels,
        activation,
        dropout,
    ):
        super().__init__()
        self.embedding = nn.Embedding(n_atom_type, hidden_channels)
        linear_rbf_layers = [
            nn.Linear(num_radial, hidden_channels),
            get_activation(activation),
        ]
        self.linear_rbf = nn.Sequential(*linear_rbf_layers)
        concat_layers = [
            nn.Linear(3 * hidden_channels, hidden_channels),
            get_activation(activation),
        ]
        if dropout:
            concat_layers.append(nn.Dropout(p=dropout))
        self.linear_concat = nn.Sequential(*concat_layers)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        self.embedding.weight.data.uniform_(-np.sqrt(3), np.sqrt(3))

        # add glorot init
        for module in [self.linear_rbf, self.linear_concat]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    glorot_orthogonal(layer.weight, scale=2.0)
                    if layer.bias is not None:
                        torch.nn.init.zeros_(layer.bias)

    def forward(
        self, x: Tensor, rbf: Tensor, i: Tensor, j: Tensor
    ) -> torch.Tensor:
        """
        Forward pass of embedding block.

        Args:
            z (torch.Tensor): Atomic numbers [N]

        Returns:
            torch.Tensor: Atom embeddings [N, emb_size]
        """
        x = self.embedding(x)
        rbf = self.linear_rbf(rbf)
        x = self.linear_concat(torch.cat([x[i], x[j], rbf], dim=-1))
        return x


class BesselBasisLayer(nn.Module):
    """
    Layer for computing Bessel basis functions.
    """

    def __init__(
        self, num_radial: int, cutoff: float, envelope_exponent: float
    ):
        super().__init__()
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        # Initialize radial basis functions
        self.freq = nn.Parameter(torch.Tensor(num_radial))

    def reset_parameters(self):
        """
        Resets the parameters of the layer.
        """
        nn.init.uniform_(self.freq, 0, 1)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        dist = dist.unsqueeze(-1) / self.cutoff
        return self.envelope(dist) * (self.freq * dist).sin()


class SphericalBasisLayer(nn.Module):
    """
    Layer for computing spherical harmonics.
    """

    def __init__(
        self,
        num_spherical: int,
        num_radial: int,
        cutoff: float,
        envelope_exponent: int,
    ):
        super().__init__()
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.envelope = Envelope(int(envelope_exponent))

        bessel_forms = bessel_basis(num_spherical, num_radial)
        sph_harm_forms = real_sph_harm(num_spherical)
        self.bessel_funcs: list[Callable] = []
        self.sph_harm_funcs: list[Callable] = []

        x, theta = sym.symbols("x theta")
        modules = {"sin": torch.sin, "cos": torch.cos}
        for i in range(num_spherical):
            if i == 0:
                sph1_val = sym.lambdify([theta], sph_harm_forms[i][0], modules)(
                    0
                )
                self.sph_harm_funcs.append(
                    lambda x, s=sph1_val: torch.zeros_like(x) + s
                )
            else:
                sph = sym.lambdify([theta], sph_harm_forms[i][0], modules)
                self.sph_harm_funcs.append(sph)
            for j in range(num_radial):
                bessel = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(bessel)

    def forward(self, dist, angle, idx_kj):
        dist = dist / self.cutoff
        rbf = torch.stack([f(dist) for f in self.bessel_funcs], dim=1)
        rbf = self.envelope(dist).unsqueeze(-1) * rbf

        cbf = torch.stack([f(angle) for f in self.sph_harm_funcs], dim=1)

        n, k = self.num_spherical, self.num_radial
        out = (rbf[idx_kj].view(-1, n, k) * cbf.view(-1, n, 1)).view(-1, n * k)
        return out


class ResidualLayer(torch.nn.Module):
    def __init__(self, hidden_channels, activation, dropout):
        super().__init__()
        layers = []
        for _ in range(2):
            layers.append(nn.Linear(hidden_channels, hidden_channels))
            layers.append(get_activation(activation))
            if dropout:
                layers.append(nn.Dropout(p=dropout))
        self.mlp = nn.Sequential(*layers)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize or reset all learnable parameters of the module."""
        # Initialize basis transformations
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                glorot_orthogonal(layer.weight, scale=2.0)
                layer.bias.data.fill_(0)

    def forward(self, x):
        return x + self.mlp(x)


class DimeNetInteractionBlock(nn.Module):
    """A neural network module that processes interactions between nodes.

    This module implements a message passing neural network block that processes
    three-way interactions between nodes in a graph, using radial and
    spherical basis functions for geometric representations.

    Attributes:
        activation: Activation function module
        hidden_channels: Number of hidden channels in the network
        int_emb_size: Size of the interaction embedding
        dropout: Dropout probability for regularization
    """

    def __init__(
        self,
        hidden_channels: int,
        int_emb_size: int,
        basis_emb_size: int,
        num_spherical: int,
        num_radial: int,
        num_before_skip: int,
        num_after_skip: int,
        activation: str,
        dropout: float,
    ):
        """Initialize the InteractionBlock.

        Args:
            hidden_channels: Number of hidden channels in the network
            int_emb_size: Size of the interaction embedding
            basis_emb_size: Size of the basis embedding
            num_spherical: Number of spherical harmonics
            num_radial: Number of radial basis functions
            num_before_skip: Number of residual layers before skip connection
            num_after_skip: Number of residual layers after skip connection
            activation: Activation function name as string
            dropout: Dropout probability
        """
        super().__init__()
        self.act = get_activation(activation)
        self.dropout = (
            nn.Dropout(p=dropout) if dropout is not None else nn.Identity()
        )

        # Basis function transformations
        self.basis_transformations = nn.ModuleDict(
            {
                "rbf": nn.Sequential(
                    nn.Linear(num_radial, basis_emb_size, bias=False),
                    nn.Linear(basis_emb_size, hidden_channels, bias=False),
                ),
                "sbf": nn.Sequential(
                    nn.Linear(
                        num_spherical * num_radial, basis_emb_size, bias=False
                    ),
                    nn.Linear(basis_emb_size, int_emb_size, bias=False),
                ),
            }
        )

        # Message transformations
        self.message_transformations = nn.ModuleDict(
            {
                "kj": nn.Linear(hidden_channels, hidden_channels),
                "ji": nn.Linear(hidden_channels, hidden_channels),
            }
        )

        # Interaction projections
        self.projections = nn.ModuleDict(
            {
                "down": nn.Linear(hidden_channels, int_emb_size, bias=False),
                "up": nn.Linear(int_emb_size, hidden_channels, bias=False),
            }
        )

        # Residual layers
        self.layers_before_skip = nn.ModuleList(
            [
                ResidualLayer(hidden_channels, activation, dropout)
                for _ in range(num_before_skip)
            ]
        )

        self.lin = nn.Linear(hidden_channels, hidden_channels)

        self.layers_after_skip = nn.ModuleList(
            [
                ResidualLayer(hidden_channels, activation, dropout)
                for _ in range(num_after_skip)
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize or reset all learnable parameters of the module."""
        # Initialize basis transformations
        for transform in self.basis_transformations.values():
            for layer in transform:
                glorot_orthogonal(layer.weight, scale=2.0)

        # Initialize message transformations
        for transform in self.message_transformations.values():
            glorot_orthogonal(transform.weight, scale=2.0)
            transform.bias.data.fill_(0)

        # Initialize projections
        for projection in self.projections.values():
            glorot_orthogonal(projection.weight, scale=2.0)

        # Initialize residual layers
        for layer in self.layers_before_skip:
            layer.reset_parameters()

        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)

        for layer in self.layers_after_skip:
            layer.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        rbf: torch.Tensor,
        sbf: torch.Tensor,
        idx_kj: torch.Tensor,
        idx_ji: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the interaction block.

        Args:
            x: Input node features
            rbf: Radial basis functions
            sbf: Spherical basis functions
            idx_kj: Indices for k->j interactions
            idx_ji: Indices for j->i interactions

        Returns:
            torch.Tensor: Updated node features
        """
        # Transform basis functions
        rbf = self.basis_transformations["rbf"](rbf)
        sbf = self.basis_transformations["sbf"](sbf)

        # Process messages
        x_ji = self.dropout(self.act(self.message_transformations["ji"](x)))
        x_kj = self.dropout(
            self.act(self.message_transformations["kj"](x)) * rbf
        )

        # Process interaction terms
        x_kj = self.act(self.projections["down"](x_kj))
        x_kj = x_kj[idx_kj] * sbf
        x_kj = self.dropout(x_kj)

        # Aggregate messages
        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x.size(0))
        x_kj = self.dropout(self.act(self.projections["up"](x_kj)))

        # Combine messages and apply residual layers
        h = x_ji + x_kj
        for layer in self.layers_before_skip:
            h = layer(h)

        h = self.dropout(self.act(self.lin(h)))
        h = h + x

        for layer in self.layers_after_skip:
            h = layer(h)

        return h


class OutputBlock(nn.Module):
    """Neural network module for processing and outputting node features.

    This module applies radial basis function transformations and multi-layer
    processing to generate final node features.

    Attributes:
        act: Activation function module
        encoder: Boolean flag for encoder mode
        dropout: Dropout layer for regularization (or Identity if no dropout)
    """

    def __init__(
        self,
        num_radial: int,
        hidden_channels: int,
        out_emb_size: int,
        out_channels: int,
        num_layers: int,
        activation: str,
        dropout: float | None = None,
    ):
        """Initialize the OutputBlock.

        Args:
            num_radial: Number of radial basis functions
            hidden_channels: Number of hidden channels
            out_emb_size: Size of the output embedding
            out_channels: Number of output channels
            num_layers: Number of processing layers
            activation: Activation function name as string
            dropout: Dropout probability (None for no dropout)
        """
        super().__init__()
        self.act = get_activation(activation)
        self.dropout = (
            nn.Dropout(p=dropout) if dropout is not None else nn.Identity()
        )

        # Basis function transformation
        self.basis_transformation = nn.Linear(
            num_radial, hidden_channels, bias=False
        )

        # Output projection
        self.output_projection = nn.Linear(
            hidden_channels, out_emb_size, bias=False
        )

        # Processing layers
        modules: list[nn.Module] = []
        for _ in range(num_layers):
            modules.append(nn.Linear(out_emb_size, out_emb_size))
            modules.append(self.act)
            if dropout is not None:
                modules.append(nn.Dropout(p=dropout))
        self.layers = nn.Sequential(*modules)
        # Final output layer
        self.final_layer = nn.Linear(out_emb_size, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize or reset all learnable parameters of the module."""
        # Initialize basis transformation
        glorot_orthogonal(self.basis_transformation.weight, scale=2.0)

        # Initialize output projection
        glorot_orthogonal(self.output_projection.weight, scale=2.0)

        # Initialize processing layers
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                glorot_orthogonal(layer.weight, scale=2.0)
                layer.bias.data.fill_(0)

        # Initialize final layer
        self.final_layer.weight.data.fill_(0)

    def forward(
        self,
        x: torch.Tensor,
        rbf: torch.Tensor,
        idx_i: torch.Tensor,
        num_nodes: int | None = None,
    ) -> torch.Tensor:
        """Forward pass of the output block.

        Args:
            x: Input node features
            rbf: Radial basis functions
            idx_i: Indices for aggregation
            num_nodes: Number of nodes (optional)

        Returns:
            torch.Tensor: Output node features
        """
        # Transform basis functions and combine with input
        basis = self.basis_transformation(rbf)
        x = basis * x

        # Aggregate features
        x = scatter(x, idx_i, dim=0, dim_size=num_nodes)

        # Project to output embedding size
        x = self.output_projection(x)

        x = self.layers(x)

        # Final output transformation
        return self.final_layer(x)


class DimeNet(nn.Module):
    """
    Directional Message Passing Neural Network (DimeNet).

    Args:
        config (DimeNetConfig): Model configuration parameters.

    References:
        .. [1] Klicpera, J., Groß, J., & Günnemann, S. (2020).
               Directional message passing for molecular graphs.
               ICLR 2020.
    """

    def __init__(self, config: DimeNetConfig):
        super().__init__()
        self.config = config

        if config.cutoff <= 0:
            raise ValueError("Cutoff radius must be positive")

        self.embedding = EmbeddingBlock(
            config.n_atom_type,
            config.num_radial,
            config.hidden_channels,
            config.activation,
            config.dropout,
        )
        self.rbf = BesselBasisLayer(
            config.num_radial,
            config.cutoff,
            int(config.envelope_exponent),
        )
        self.sbf = SphericalBasisLayer(
            config.num_spherical,
            config.num_radial,
            config.cutoff,
            int(config.envelope_exponent),
        )
        # Interaction blocks
        self.interaction_blocks = nn.ModuleList(
            [
                DimeNetInteractionBlock(
                    config.hidden_channels,
                    config.int_emb_size,
                    config.basis_emb_size,
                    config.num_spherical,
                    config.num_radial,
                    config.num_before_skip,
                    config.num_after_skip,
                    config.activation,
                    (
                        float(config.dropout)
                        if config.dropout is not None
                        else 0.0
                    ),
                )
                for _ in range(config.num_blocks)
            ]
        )

        # Output network
        self.output_blocks = nn.ModuleList(
            [
                OutputBlock(
                    config.num_radial,
                    config.hidden_channels,
                    config.out_emb_size,
                    config.out_channels,
                    config.num_output_layers,
                    config.activation,
                    config.dropout,
                )
                for _ in range(config.num_blocks + 1)
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize model parameters using uniform Xavier initialization."""
        self.embedding.reset_parameters()
        self.rbf.reset_parameters()
        for output_block in self.output_blocks:
            output_block.reset_parameters()
        for interaction_block in self.interaction_blocks:
            interaction_block.reset_parameters()

    def forward(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of DimeNet.

        Args:
            z (torch.Tensor): Atomic numbers [N]
            pos (torch.Tensor): Atomic positions [N, 3]
            batch (torch.Tensor): Batch assignments [N]

        Returns:
            torch.Tensor: Predicted properties [batch_size, output_dim]
        """
        edge_index, edge_length = radius_graph(
            pos, batch, self.config.cutoff, self.config.max_num_neighbors
        )

        dst, src, idx_i, idx_j, idx_k, idx_kj, idx_ji = find_triplets(
            edge_index, num_nodes=z.size(0)
        )
        # Compute angles between triplets using cross product / dot product
        pos_ji, pos_kj = pos[idx_ji] - pos[idx_i], pos[idx_kj] - pos[idx_i]
        a = (pos_ji * pos_kj).norm(dim=-1)
        b = torch.cross(pos_ji, pos_kj).norm(dim=-1)
        angles = torch.atan2(b, a)

        # Distance and angles encoding
        rbf = self.rbf(edge_length)
        sbf = self.sbf(edge_length, angles, idx_kj)

        # Initial embedding, and collect the output from the embedding block
        x = self.embedding(z, rbf, dst, src)
        out = self.output_blocks[0](x, rbf, dst, num_nodes=pos.size(0))

        # Interaction blocks followed by output collection blocks
        for interaction_block, output_block in zip(
            self.interaction_blocks, self.output_blocks[1:], strict=False
        ):
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
            out = torch.add(out + output_block(x, rbf, dst))

        # Global pooling
        out = torch.zeros(
            batch.max().item() + 1, x.size(-1), device=x.device, dtype=x.dtype
        )
        out.index_add_(0, batch, x)

        return out
