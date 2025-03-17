"""Graph neural network utility functions.

This module provides utility functions and classes for graph neural networks,
including activation functions, graph construction, and triplet finding.
"""

from typing import Literal

import torch
from torch import Tensor, nn
from torch_sparse import SparseTensor  # type: ignore


class ShiftedSoftplus(nn.Module):
    """Shifted Softplus activation function: ln(0.5e^x + 0.5)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of shifted softplus.

        Args:
            x: Input tensor

        Returns:
            Activated tensor
        """
        return torch.log(0.5 * torch.exp(x) + 0.5)


def get_activation(
    activation: Literal["relu", "gelu", "silu", "tanh", "ssp"],
) -> nn.Module:
    """Get activation function from string.

    Args:
        activation: Name of the activation function

    Returns:
        Activation module

    Raises:
        KeyError: If activation function is not supported
    """
    activations = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
        "tanh": nn.Tanh(),
        "ssp": ShiftedSoftplus(),
    }
    return activations[activation.lower()]


def radius_graph(
    pos: torch.Tensor,
    batch: torch.Tensor,
    cutoff: float,
    max_num_neighbors: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a radius graph from atomic positions.

    Args:
        pos: Position tensor [N, 3]
        batch: Batch tensor [N]
        cutoff: Max distance for edges
        max_num_neighbors: Max neighbors per atom

    Returns:
        Tuple containing:
            - edge_index [2, E]: Source and destination node indices
            - edge_attr [E]: Edge distances
    """
    if pos.size(0) == 0:  # Handle empty batch
        return (
            torch.empty((2, 0), dtype=torch.long, device=pos.device),
            torch.empty(0, device=pos.device),
        )

    dist = torch.cdist(pos, pos)
    src, dst = torch.where(dist < cutoff + 1e-6)

    # Filter out self-loops
    mask = dist[src, dst] != 0
    src, dst = src[mask], dst[mask]
    edge_attr = dist[src, dst]

    # Filter out edges between different graphs
    mask = batch[src] == batch[dst]
    src, dst = src[mask], dst[mask]
    edge_attr = edge_attr[mask]

    keep_indices = []
    if max_num_neighbors is not None:
        # For each target node, sort neighbors by distance and keep top k
        unique_src = torch.unique(src)
        for node in unique_src:
            node_mask = src == node
            node_indices = torch.where(node_mask)[0]
            n_neighbors = len(node_indices)

            # If we have fewer than k neighbors, keep all
            if n_neighbors <= max_num_neighbors:
                keep_indices.append(node_indices)
                continue

            # Sort neighbors by distance and keep top k
            node_attr = edge_attr[node_mask]
            _, top_k = torch.topk(node_attr, k=max_num_neighbors, largest=False)
            keep_indices.append(node_indices[top_k])

        keep_indices = torch.cat(keep_indices)
        src = src[keep_indices]
        dst = dst[keep_indices]
        edge_attr = edge_attr[keep_indices]

    edge_index = torch.stack([src, dst], dim=0)
    return edge_index, edge_attr


def find_triplets(edge_index: Tensor, num_nodes: int) -> tuple[Tensor, ...]:
    """Find triangular patterns (k->j->i) in a graph.

    Args:
        edge_index: Edge indices representing directed edges (j->i)
        num_nodes: Total number of nodes in the graph

    Returns:
        Tuple containing:
            - dst: Destination nodes
            - src: Source nodes
            - idx_i: Target nodes in triplets
            - idx_j: Middle nodes in triplets
            - idx_k: Source nodes in triplets
            - idx_kj: Edge indices for k->j edges
            - idx_ji: Edge indices for j->i edges
    """
    src, dst = edge_index  # Rename for clarity: source->destination edges

    # Create sparse adjacency matrix
    edge_ids = torch.arange(src.size(0), device=src.device)
    sparse_size = (num_nodes, num_nodes)
    adj = SparseTensor(
        row=dst,
        col=src,
        value=edge_ids,
        sparse_sizes=sparse_size,
    )

    # Find neighboring nodes for each source node
    neighbors = adj[src]
    triplet_counts = neighbors.set_value(None).sum(dim=1).long()

    # Generate triplet node indices
    idx_i = dst.repeat_interleave(triplet_counts)
    idx_j = src.repeat_interleave(triplet_counts)
    idx_k = neighbors.storage.col()

    # Remove self-loops
    valid_triplets = idx_i != idx_k
    idx_i, idx_j, idx_k = (idx[valid_triplets] for idx in (idx_i, idx_j, idx_k))

    # Get corresponding edge indices
    idx_kj = neighbors.storage.value()[valid_triplets]
    idx_ji = neighbors.storage.row()[valid_triplets]

    return dst, src, idx_i, idx_j, idx_k, idx_kj, idx_ji
