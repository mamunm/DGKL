"""Ensemble model training with hyperparameters optimization."""

import torch
from sklearn.metrics import r2_score
from cat_uncertainty.dgkl.train_schnet_atomic import train_schnet
from cat_uncertainty.utils.console import Console
import gpytorch
from torch_scatter import scatter

#######################Experiment and logger parameters##############################
logger = Console("dgkl_training.log")
MAX_EPOCHS = 200
SEED = 107474
DATA = "eagle"
BATCH_SIZE = 64
# SchNet
HIDDEN_CHANNELS = 1024
NUM_FILTERS = 128
NUM_INTERACTIONS = 6
CFCONV_NUM_DENSE = 2
INTERACTION_NUM_DENSE = 2
NUM_GAUSSIAN = 50
READOUT = "atomic"
OUTPUT_DIM = 64
ACTIVATION = "ssp"
DROPOUT = None
LAYER_NORM = True
ELEMENTWISE_AFFINE = False
# Optimizer and scheduler
OPTIMIZER = "adam"
SCHEDULER = "reduce_on_plateau"
LEARNING_RATE = 1e-2
# svgp
KERNEL_TYPE = "rbf"
DIST_TYPE = "cholesky"
VARIATIONAL_STRATEGY = "standard"
INDUCING_TYPE = "random"
NUM_INDUCING = 512
MLL = "PLL"
GRID_BOUNDS = False
MAX_GRAD_NORM = None
# Project and name
PROJECT = "hidden_channels_1024"
NAME = "seed_107474"

logger.print("Strat ensemble training...")
logger.print(f"PyTorch Version: {torch.__version__}")
logger.print(f"Max epochs: {MAX_EPOCHS}, Seed: {SEED}")

config = {
    "batch_size": BATCH_SIZE,
    "hidden_channels": HIDDEN_CHANNELS,
    "num_filters": NUM_FILTERS,
    "num_interactions": NUM_INTERACTIONS,
    "cfconv_num_dense": CFCONV_NUM_DENSE,
    "interaction_num_dense": INTERACTION_NUM_DENSE,
    "num_gaussians": NUM_GAUSSIAN,
    "readout": READOUT,
    "output_dim": OUTPUT_DIM,
    "activation": ACTIVATION,
    "dropout": DROPOUT,
    "layer_norm": LAYER_NORM,
    "elementwise_affine": ELEMENTWISE_AFFINE,
    "optimizer": OPTIMIZER,
    "scheduler": SCHEDULER,
    "learning_rate": LEARNING_RATE,
    "kernel_type": KERNEL_TYPE,
    "dist_type": DIST_TYPE,
    "variational_strategy": VARIATIONAL_STRATEGY,
    "inducing_type": INDUCING_TYPE,
    "num_inducing": NUM_INDUCING,
    "mll": MLL,
    "grid_bounds": GRID_BOUNDS,
    "max_grad_norm": MAX_GRAD_NORM,
}

feature_extractor, svgp, likelihood, train_dataloader, val_dataloader, test_dataloader = train_schnet(
        config=config,
        data=DATA, 
        logger=logger, 
        max_epochs=MAX_EPOCHS,
        seed=SEED, 
        project=PROJECT, 
        name=NAME
)

feature_extractor.eval()
svgp.eval()
likelihood.eval()

if config["grid_bounds"]:
    grid_bounds = gpytorch.utils.grid.ScaleToBounds(-1, 1)
else:
    grid_bounds = None

predictions = {}
metrics = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for split in ["train", "val", "test"]:
    logger.print(f"\nEvaluating {split} set...")

    if split == "train":
        loader = train_dataloader
    elif split == "val":
        loader = val_dataloader
    else:
        loader = test_dataloader

    y_true_list = []
    y_pred_list = []
    y_std_list = []
    latent_features = []

    with torch.no_grad():
        for batch in loader:
            z = batch["atomic_numbers"].long().to(device)
            pos = batch["pos"].to(device)
            graph_batch = batch["batch"].to(device)
            y_true = batch["y_relaxed"].to(device)

            features = feature_extractor(z, pos, graph_batch)
            if grid_bounds:
                features = grid_bounds(features)
            try:
                output = likelihood(svgp(features))
            except ValueError:
                continue
            mean, covar = output.mean, output.covariance_matrix
            mean_scatter = scatter(mean, graph_batch, 0, reduce="sum")
            covar_scatter = torch.eye(max(graph_batch) + 1).to(covar) * scatter(
                covar.diag(), graph_batch, 0, reduce="sum"
            )
            output = gpytorch.distributions.MultivariateNormal(
                mean_scatter, covar_scatter
            )
            y_pred = output.mean
            y_std = output.stddev

            y_true_list.append(y_true.cpu())
            y_pred_list.append(y_pred.cpu())
            y_std_list.append(y_std.cpu())
            latent_features.append(features.cpu())

    y_true = torch.cat(y_true_list).view(-1)
    y_pred = torch.cat(y_pred_list).view(-1)
    y_std = torch.cat(y_std_list).view(-1)
    latent_features = torch.cat(latent_features)

    predictions[split] = {
        "y_true": y_true, 
        "y_pred": y_pred, 
        "y_std": y_std, 
        "latent_features": latent_features
    }

    r2 = r2_score(y_true.numpy(), y_pred.numpy())
    rmse = torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()
    metrics[split] = {"R²": r2, "RMSE": rmse}

    logger.print(f"• R²: {r2:.4f}")
    logger.print(f"• RMSE: {rmse:.4f}")

results = {"predictions": predictions, "metrics": metrics}
torch.save(results, "results.pt")
logger.print("\n✓ Results saved to results.pt!")

