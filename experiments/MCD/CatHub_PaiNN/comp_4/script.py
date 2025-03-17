"""Monte Carlo Dropout model training with hyperparameters optimization."""

import os
import torch
from pathlib import Path
from functools import partial
from sklearn.metrics import r2_score
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from cat_uncertainty.graph_models.train_painn import train_painn
from cat_uncertainty.utils.console import Console
from cat_uncertainty.utils.utils import suppress_output
from cat_uncertainty.monte_carlo_model.mcd_model import MCDropout

#######################Experiment and logger parameters##############################
logger = Console("mcd_training.log")
MAX_EPOCHS = 200
SEED = 29197
HP_MAX_EPOCHS = 25
HP_NUM_RUN = 25
DATA = "cathub"

logger.print("Strat ensemble training...")
logger.print(f"PyTorch Version: {torch.__version__}")
logger.print(f"Max epochs: {MAX_EPOCHS}, Seed: {SEED}")

#######################Hyperparameters################################################
hyperparameters = {
    "batch_size": tune.choice([32, 64]),
    "num_layers": tune.choice([2, 3, 4, 5, 6]),
    "hidden_channels": tune.choice([32, 64, 128, 256]),
    "num_radial": tune.choice([25, 50, 75]),
    "readout": tune.choice(["add", "mean"]),
    "dropout": tune.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
    "activation": tune.choice(["ssp", "relu", "tanh", "gelu", "silu"]),
    "optimizer": tune.choice(["adam", "adamw", "sgd", "rmsprop"]),
    "scheduler": tune.choice(["cosine", "reduce_on_plateau", "step"]),
    "learning_rate": tune.loguniform(1e-5, 1e-3),
}

scheduler = ASHAScheduler(
    max_t=HP_MAX_EPOCHS,
    grace_period=min(HP_MAX_EPOCHS // 2, 10),
    reduction_factor=2,
)
searcher = OptunaSearch()
ray_path = Path("ray_results").resolve()
ray_path.mkdir(parents=True, exist_ok=True)

hp_train_painn = partial(
    train_painn,
    data=DATA,
    max_epochs=HP_MAX_EPOCHS,
    checkpoint_dir="hp_checkpoints",
    seed=SEED,
    project="hp",
    name="hp",
    htune=True,
)
logger.print("Hyperparameter tuning started...")
with suppress_output():
    analysis = tune.run(
        hp_train_painn,
        config=hyperparameters,
        scheduler=scheduler,
        search_alg=searcher,
        num_samples=HP_NUM_RUN,
        resources_per_trial={"cpu": 4, "gpu": 1},
        metric="val_loss",
        mode="min",
        max_failures=5,
        storage_path=str(ray_path),
)
logger.print("Hyperparameter tuning finished.")

best_config = analysis.best_config
logger.print("Best config: ", best_config)


data_module, pl_module = train_painn(
    config=best_config,
    data=DATA, 
    logger=logger, 
    max_epochs=MAX_EPOCHS,
    checkpoint_dir="checkpoints", 
    seed=SEED, 
    project="mcd_painn_cathub", 
    name="run_29197",
    htune=False
)

mcd = MCDropout(pl_module.model, num_forward_passes=10)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mcd = mcd.to(device)

predictions, metrics = {}, {}

for split in ["train", "val", "test"]:
    logger.print(f"Evaluating {split} set...")
    if split == "train":
        loader = data_module.train_dataloader()
    elif split == "val":
        loader = data_module.val_dataloader()
    else:
        loader = data_module.test_dataloader()
    
    y_true_list = []
    y_pred_list = []
    y_std_list = []

    mcd.eval()
    def enable_dropout(m: torch.nn.Module):
        if isinstance(m, torch.nn.Dropout):
            m.train()
    mcd.apply(enable_dropout)
    with torch.no_grad():
        for batch in loader:
            z = batch["atomic_numbers"].long().to(device)
            pos = batch["pos"].to(device)
            graph_batch = batch["batch"].to(device)
            y_true = batch["y_relaxed"].to(device)
            y_pred, y_std = mcd(z, pos, graph_batch)
            y_true_list.append(y_true.cpu())
            y_pred_list.append(y_pred.cpu())
            y_std_list.append(y_std.cpu())

    y_true = torch.cat(y_true_list).view(-1)
    y_pred = torch.cat(y_pred_list).view(-1)
    y_std = torch.cat(y_std_list).view(-1)

    predictions[split] = {"y_true": y_true, "y_pred": y_pred, "y_std": y_std}

    r2 = r2_score(y_true.numpy(), y_pred.numpy())
    rmse = torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()
    metrics[split] = {"R²": r2, "RMSE": rmse}

    logger.print(f"R²: {r2:.4f} | RMSE: {rmse:.4f}")

results = {"predictions": predictions, "metrics": metrics}
torch.save(results, "results.pt")
logger.print("✓ Predictions saved to results.pt!")




