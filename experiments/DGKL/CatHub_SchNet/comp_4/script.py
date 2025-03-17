"""Ensemble model training with hyperparameters optimization."""

import torch
from sklearn.metrics import r2_score
from cat_uncertainty.dgkl.train_schnet import train_schnet
from cat_uncertainty.utils.console import Console
import gpytorch

#######################Experiment and logger parameters##############################
logger = Console("dgkl_training.log")
MAX_EPOCHS = 400
SEED = 229259
DATA = "cathub"
BATCH_SIZE = 64
# SchNet
HIDDEN_CHANNELS = 256
NUM_FILTERS = 128
NUM_INTERACTIONS = 6
CFCONV_NUM_DENSE = 2
INTERACTION_NUM_DENSE = 2
NUM_GAUSSIAN = 50
READOUT = "add"
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
GRID_BOUNDS = True
MAX_GRAD_NORM = 20.0
# Project and name
PROJECT = "output_dim_64"
NAME = "seed_229259"

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
            output = likelihood(svgp(features))
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

