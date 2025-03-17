"""Ensemble model training with hyperparameters optimization."""

import os
import torch
from pathlib import Path
from functools import partial
from sklearn.metrics import r2_score
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from cat_uncertainty.graph_models.train_schnet import train_schnet
from cat_uncertainty.utils.console import Console
from cat_uncertainty.utils.utils import suppress_output
from cat_uncertainty.ensemble_model.ensemble import EnsembleModel

#######################Experiment and logger parameters##############################
logger = Console("ensemble_training.log")
MAX_EPOCHS = 100
SEED = 48094
TOPK = 5
HP_MAX_EPOCHS = 25
HP_NUM_RUN = 25
DATA = "cathub"

logger.print("Strat ensemble training...")
logger.print(f"PyTorch Version: {torch.__version__}")
logger.print(f"Max epochs: {MAX_EPOCHS}, Seed: {SEED}")

#######################Hyperparameters################################################
hyperparameters = {
    "batch_size": tune.choice([32, 64]),
    "hidden_channels": tune.choice([32, 64, 128, 256]),
    "num_filters": tune.choice([32, 64, 128, 256]),
    "num_interactions": tune.choice([2, 4, 6]),
    "cfconv_num_dense": tune.choice([2, 4, 6]),
    "interaction_num_dense": tune.choice([2, 4, 6]),
    "num_gaussians": tune.choice([25, 50, 75]),
    "readout": tune.choice(["add", "mean"]),
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

hp_train_schnet = partial(
    train_schnet,
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
        hp_train_schnet,
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
df = analysis.results_df
sorted_df = df.sort_values("val_loss", ascending=True)
top_k_configs = sorted_df.head(TOPK)

top_k_configs.to_csv("top_k_configs.csv", index=False)
logger.print("Top K configurations saved to top_k_configs.csv")

ensemble_models = []
for i, (_, row) in enumerate(top_k_configs.iterrows()):
    logger.print(f"Training final model {i+1}/{TOPK}")
    config = {
        k.split("/")[-1]: v
        for k, v in row.items()
        if k.split("/")[-1] in best_config
    }
    data_module, pl_module = train_schnet(
        config=config,
        data=DATA, 
        logger=logger, 
        max_epochs=MAX_EPOCHS,
        checkpoint_dir=f"checkpoints_{i+1}", 
        seed=SEED, 
        project="ensemble_schnet_cathub", 
        name="run_48094",
        htune=False
    )
    ensemble_models.append(pl_module)

ensemble = EnsembleModel(models=ensemble_models)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ensemble = ensemble.to(device)

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

    ensemble.eval()
    with torch.no_grad():
        for batch in loader:
            z = batch["atomic_numbers"].long().to(device)
            pos = batch["pos"].to(device)
            graph_batch = batch["batch"].to(device)
            y_true = batch["y_relaxed"].to(device)
            y_pred, y_std = ensemble(z, pos, graph_batch)
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




