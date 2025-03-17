"""Inducing point initialization using random sampling."""


import torch
from torch import nn


def random_inducing_points(
    dataloader: torch.utils.data.DataLoader,
    feature_extractor: nn.Module | None = None,
    device: torch.device | None = None,
    num_inducing: int | None = None,
    use_x: bool = False,
    use_batch: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Initialize inducing points by randomly sampling from the dataset.

    Args:
        dataloader (torch.utils.data.DataLoader): Train data loader
        feature_extractor (nn.Module | None): Feature extractor module
        device (torch.device | None): Device to use for computation
        num_inducing (int | None): Number of inducing points to use
        use_x (bool): Whether to use the x attribute of the data
        use_batch (bool): Whether to use the batch attribute of the data

    Returns:
        Tuple containing:
            - torch.Tensor: Tensor of inducing points
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
    total_batches = len(dataloader)
    for i, batch in enumerate(dataloader):
        print(f"Batch {i+1}/{total_batches}", end="\r")
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
        all_features.append(latent_features.cpu())
        all_targets.append(batch["y_relaxed"].cpu())

    features = torch.cat(all_features, dim=0)
    targets = torch.cat(all_targets, dim=0)

    num_total = features.size(0)
    indices = torch.randperm(num_total)[:num_inducing]
    inducing_points = features[indices].to(device=device)

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
