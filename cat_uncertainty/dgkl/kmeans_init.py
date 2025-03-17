"""Inducing point initialization using KMeans clustering."""


import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import nn


def kmeans_inducing_points(
    dataloader: torch.utils.data.DataLoader,
    feature_extractor: nn.Module | None = None,
    device: torch.device | None = None,
    num_inducing: int | None = None,
    use_x: bool = False,
    use_batch: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Initialize inducing points using KMeans or MiniBatch K-means clustering.

    Args:
        dataloader (torch.utils.data.DataLoader): Train data loader
        feature_extractor (nn.Module | None): Feature extractor module
        device (torch.device | None): Device to use for computation
        num_inducing (int | None): Number of inducing points to use
        use_x (bool): Whether to use the x attribute of the data
        use_batch (bool): Whether to use the batch attribute of the data

    Returns:
        Tuple containing:
            - torch.Tensor: Tensor of inducing points (cluster centroids)
            - Dict[str, float]: Target scaling parameters
    """
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not num_inducing:
        num_inducing = 100

    if feature_extractor is not None:
        feature_extractor = feature_extractor.to(
            device=device, dtype=torch.float
        )

    all_features = []
    all_targets = []
    for batch in dataloader:
        if use_x:
            x = batch["x"].to(device=device, dtype=torch.float)
            edge_index = batch["edge_index"].to(device=device, dtype=torch.long)
            graph_batch = batch["batch"].to(device=device, dtype=torch.long)
            features = (x, edge_index, graph_batch)
        elif use_batch:
            features = (batch.to(device),)
        else:
            z = batch["atomic_numbers"].to(device=device, dtype=torch.long)
            pos = batch["pos"].float().to(device=device, dtype=torch.float)
            graph_batch = batch["batch"].to(device=device, dtype=torch.long)
            features = (z, pos, graph_batch)

        if feature_extractor is not None:
            with torch.no_grad():
                latent_features = feature_extractor(*features)
        else:
            latent_features = features[0]
        all_features.append(latent_features.cpu().numpy())
        all_targets.append(batch["y_relaxed"].cpu())

    features_array = np.concatenate(all_features, axis=0)
    targets = torch.cat(all_targets, dim=0)

    kmeans = KMeans(
        n_clusters=num_inducing,
        n_init=10,
        random_state=42,
    )
    kmeans.fit(features_array)

    inducing_points = torch.tensor(
        kmeans.cluster_centers_, dtype=torch.float, device=device
    )

    target_mean = targets.mean().item()
    target_std = targets.std().item()
    target_min = targets.min().item()
    target_max = targets.max().item()

    scale_params = {
        "min": float(target_min),
        "max": float(target_max),
        "mean": float(target_mean),
        "std": float(target_std),
    }

    return inducing_points, scale_params
