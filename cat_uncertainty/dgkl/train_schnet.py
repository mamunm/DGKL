"""Script to train a SchNet model on a dataset using DGKL."""
import os
import time
import torch
import gpytorch

import cat_uncertainty
from ..data.config import (
    DataLoaderParams,
    DataModuleConfig,
    DBParams,
)
from ..data.datamodule import GraphDataModule
from typing import Dict, Any, Literal

from ..utils.console import Console
from ..graph_models.models.schnet import SchNet, SchNetModelParams
from pathlib import Path
import wandb
from sklearn.metrics import r2_score

from .dgkl import SVGP
from .random_init import random_inducing_points
from .kmeans_init import kmeans_inducing_points

#######################OS and Torch configuration####################################
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["WANDB_SILENT"] = "true"

torch.set_float32_matmul_precision("medium")
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
#####################################################################################

DATA_DICT = {
    "cathub": "catalysis_hub",
    "eagle": "open_catalyst_20/eagle_split",
}
LMDB_FILES = {
    "cathub": "CatHub.lmdb",
    "eagle": "eagle_data.lmdb",
}

def train_schnet(
    config: Dict[str, Any],
    logger: Console,
    data: Literal["cathub", "eagle"] = "cathub",
    max_epochs: int = 200,
    checkpoint_dir: str = "checkpoints",
    seed: int = 42,
    project: str = "test",
    name: str = "test",
):
    """Train a SchNet model."""
    wandb.init(
        project=project,
        name=name,
    )
    Path(checkpoint_dir).mkdir(exist_ok=True)
    #######################Data Setup####################################################
    DB_PATH = (
        Path(cat_uncertainty.__path__[0]).parent
        / f"data/{DATA_DICT[data]}"
    )
    logger.print(f"Database path: {DB_PATH}")
    db_params = DBParams(
        db_path=DB_PATH,
        lmdb_files=LMDB_FILES[data],
        train_fraction=0.70,
        val_fraction=0.10,
        test_fraction=0.20
    )
    dataloader_params = DataLoaderParams(
        batch_size=config.get("batch_size", 64),
        num_workers=4,
        train_shuffle=True,
        val_shuffle=False,
        test_shuffle=False,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=1,
        seed=seed,
    )

    data_module_config = DataModuleConfig(
        data_params=db_params,
        dataloader_params=dataloader_params,
        transform=None,
    )

    data_module = GraphDataModule(data_module_config)
    logger.print("Setting up data module...")
    data_module.setup()
    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()
    test_dataloader = data_module.test_dataloader()
    logger.print("Data module setup complete!")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    schnet_params = SchNetModelParams(
        hidden_channels=config.get("hidden_channels", 128),
        num_filters=config.get("num_filters", 128),
        num_interactions=config.get("num_interactions", 6),
        cfconv_num_dense=config.get("cfconv_num_dense", 2),
        interaction_num_dense=config.get("interaction_num_dense", 2),
        num_gaussians=config.get("num_gaussians", 50),
        cutoff=10,
        max_num_neighbors=32,
        n_atom_type=100,
        output_dim=config.get("output_dim", 64),
        activation=config.get("activation", "ssp"),
        readout=config.get("readout", "mean"),
        dropout=config.get("dropout", None),
    )
    feature_extractor = SchNet(config=schnet_params)
    if config.get("layer_norm", False):
        feature_extractor.output_net.append(
            torch.nn.LayerNorm(
                config.get("output_dim", 64),
                elementwise_affine=config.get("elementwise_affine", False))
        )
    feature_extractor = feature_extractor.to(device)
    logger.print("Feature extractor created!")
    
    inducing_fn = random_inducing_points if config.get("inducting_type", "random") == "random" else kmeans_inducing_points
    inducing_points, _ = inducing_fn(
        train_dataloader,
        feature_extractor,
        device,
        num_inducing=config.get("num_inducing", 256),
    )
    logger.print("Inducing points created!")
    
    svgp = SVGP(
        inducing_points,
        kernel_type="rbf",
        dist_type="cholesky",
        variational_strategy="standard",
    ).to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

    # Optimizer
    optimizer_fn = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
        "adamw": torch.optim.AdamW,
        "rmsprop": torch.optim.RMSprop,
    }
    opt_kwargs = {
        "adam": {"weight_decay": 1e-5},
        "sgd": {"weight_decay": 1e-5, "momentum": 0.9, "nesterov": True},
        "adamw": {"weight_decay": 1e-5},
        "rmsprop": {"weight_decay": 1e-5, "momentum": 0.9},
    }
    learning_rate = config.get("learning_rate", 1e-2)
    fe_learning_rate = learning_rate*1e-3
    optimizer = optimizer_fn[config.get("optimizer", "adam")](
        [
            {"params": feature_extractor.parameters(), "lr": fe_learning_rate},
            {"params": svgp.parameters(), "lr": learning_rate},
            {"params": likelihood.parameters(), "lr": learning_rate},
        ],
        **opt_kwargs[config.get("optimizer", "adam")]
    )
    logger.print("Optimizer created!")

    scheduler_name = config.get("scheduler", "reduce_lr_on_plateau")
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    elif scheduler_name == "cosine_warm_restarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max_epochs//8,
            T_mult=2,
            eta_min=1e-6,
        )
    elif scheduler_name == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        )
    elif scheduler_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.1,
        )
    elif scheduler_name == "fairchem":
        linear_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.001,
            end_factor=1.0,
            total_iters=5,
        )
        step_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=15,
            gamma=0.1,
        )
        scheduler = ChainedScheduler(
            [step_scheduler, linear_scheduler],
            optimizer=optimizer,
        )
    elif scheduler_name == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=learning_rate, 
            total_steps=max_epochs * len(train_dataloader), 
            pct_start=0.3,
            anneal_strategy='cos', 
            cycle_momentum=True, 
            base_momentum=0.85,
            max_momentum=0.95,
            div_factor=25., 
            final_div_factor=1e4
        )
    else:
        raise ValueError(f"Scheduler {scheduler_name} not recognized!")
    
    logger.print("Scheduler created!")
    
    if config.get("mll", "PLL") == "ELBO":
        mll = gpytorch.mlls.VariationalELBO(
            likelihood, svgp, num_data=len(train_dataloader.dataset)
        )
    else:
        mll = gpytorch.mlls.PredictiveLogLikelihood(
            likelihood, svgp, num_data=len(train_dataloader.dataset)
        )
    
    if config.get("grid_bounds", False):
        grid_bounds = gpytorch.utils.grid.ScaleToBounds(-1, 1)
    else:
        grid_bounds = None
    
    grad_norm = config.get("max_grad_norm", None)
    
    best_val_loss = float("inf")
    best_val_r2 = float("-inf")
    best_model_path = None
    early_stopping_counter = 0
    early_stopping_patience = 5
    min_delta = (
        1e-5
    )
    logger.print("Starting training...")
    for epoch in range(max_epochs):

        epoch_train_preds = []
        epoch_train_targets = []
        epoch_val_preds = []
        epoch_val_targets = []

        feature_extractor.train()
        svgp.train()
        likelihood.train()
        running_loss = 0.0

        train_start_time = time.time()
        num_train_batches = len(train_dataloader)
        for batch_idx, batch in enumerate(train_dataloader, 1):
            optimizer.zero_grad()
            z = batch["atomic_numbers"].long().to(device)
            pos = batch["pos"].to(device)
            graph_batch = batch["batch"].to(device)
            energies = batch["y_relaxed"].to(device)
            features = feature_extractor(z, pos, graph_batch)
            if grid_bounds:
                features = grid_bounds(features)
            with gpytorch.settings.cholesky_jitter(1e-3, 1e-3):
                output = svgp(features)
            try:
                loss = -mll(output, energies)
                loss.backward()
                if grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        [p for group in optimizer.param_groups for p in group["params"]],
                        max_norm=grad_norm,
                    )
                optimizer.step()
                running_loss += loss.item()

                with torch.no_grad():
                    preds = output.mean.cpu().numpy()
                    targets = energies.cpu().numpy()
                    epoch_train_preds.extend(preds)
                    epoch_train_targets.extend(targets)

            except ValueError:
                logger.print("Found a ValueError, restoring to previous best model")
                temp_models = torch.load(best_model_path)
                feature_extractor.load_state_dict(temp_models["feature_extractor"].state_dict())
                svgp.load_state_dict(temp_models["svgp"].state_dict())
                likelihood.load_state_dict(temp_models["likelihood"].state_dict())
                feature_extractor.train()
                svgp.train()
                likelihood.train()
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= 0.5

        train_time = time.time() - train_start_time
        train_iter_per_sec = num_train_batches / train_time

        avg_loss_train = running_loss / num_train_batches
        r2_train = r2_score(epoch_train_targets, epoch_train_preds)

        val_start_time = time.time()
        num_val_batches = len(val_dataloader)
        feature_extractor.eval()
        svgp.eval()
        likelihood.eval()
        running_loss = 0.0
        for batch_idx, batch in enumerate(val_dataloader, 1):
            z = batch["atomic_numbers"].long().to(device)
            pos = batch["pos"].to(device)
            graph_batch = batch["batch"].to(device)
            energies = batch["y_relaxed"].to(device)
            features = feature_extractor(z, pos, graph_batch)
            if grid_bounds:
                features = grid_bounds(features)
            output = svgp(features)
            try:
                loss = -mll(output, energies)
                running_loss += loss.item()

                with torch.no_grad():
                    preds = output.mean.cpu().numpy()
                    targets = energies.cpu().numpy()
                    epoch_val_preds.extend(preds)
                    epoch_val_targets.extend(targets)

            except ValueError:
                logger.print("Found a ValueError, restoring to previous best state.")
                temp_models = torch.load(best_model_path)
                feature_extractor.load_state_dict(temp_models["feature_extractor"].state_dict())
                svgp.load_state_dict(temp_models["svgp"].state_dict())
                likelihood.load_state_dict(temp_models["likelihood"].state_dict())
                feature_extractor.eval()
                svgp.eval()
                likelihood.eval()
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= 0.5

        val_time = time.time() - val_start_time
        val_iter_per_sec = num_val_batches / val_time

        avg_loss_val = running_loss / num_val_batches
        r2_val = r2_score(epoch_val_targets, epoch_val_preds)

        if scheduler_name == "reduce_on_plateau": 
            scheduler.step(avg_loss_val)
        else:
            scheduler.step()
        current_lr_feature_extractor = optimizer.param_groups[0]["lr"]
        current_lr_svgp = optimizer.param_groups[1]["lr"]
        current_lr_likelihood = optimizer.param_groups[2]["lr"]

        logger.print(
            f"Epoch: {epoch} |"
            f"Train Loss: {avg_loss_train:.4f} |"
            f"Train R²: {r2_train:.4f} |"
            f"Train Speed: {train_iter_per_sec:.2f} iter/s |"
            f"Val Loss: {avg_loss_val:.4f} |"
            f"Val R²: {r2_val:.4f} |"
            f"Val Speed: {val_iter_per_sec:.2f} iter/s |"
            f"FE LR: {current_lr_feature_extractor:.2e} |"
            f"SVGP LR: {current_lr_svgp:.2e} |"
            f"Likelihood LR: {current_lr_likelihood:.2e}"
        )

        wandb.log(
            {
                "train_loss": avg_loss_train,
                "train_r2": r2_train,
                "val_loss": avg_loss_val,
                "val_r2": r2_val,
                "feature_extractor_lr": current_lr_feature_extractor,
                "svgp_lr": current_lr_svgp,
                "likelihood_lr": current_lr_likelihood,
                "train_speed": train_iter_per_sec,
                "val_speed": val_iter_per_sec,
            }
        )

        if epoch == 0 or avg_loss_val < best_val_loss - min_delta:
            early_stopping_counter = 0
            best_val_loss = avg_loss_val
            best_val_r2 = r2_val

            if best_model_path is not None and os.path.exists(best_model_path):
                os.remove(best_model_path)

            best_model_path = f"{checkpoint_dir}/best_model_epoch="
            best_model_path += f"{epoch}_val_loss={best_val_loss:.4f}.pt"
            if logger:
                logger.print(
                    f"Saving best model to {best_model_path}..."
                )
            torch.save(
                {
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                    "val_r2": best_val_r2,
                    "feature_extractor": feature_extractor,
                    "svgp": svgp,
                    "likelihood": likelihood,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                },
                best_model_path,
            )
            logger.print("★ New best validation metrics!")
        else:
            early_stopping_counter += 1
            logger.print(
                f"Early stopping counter: {early_stopping_counter}/"
                f"{early_stopping_patience} (Best val_loss: "
                f"{best_val_loss:.4f})"
            )

            if early_stopping_counter >= early_stopping_patience:
                logger.print(
                    "\nEarly stopping triggered! No improvement in "
                    f"validation loss for {early_stopping_patience} "
                    "epochs."
                )
                break

    if epoch < max_epochs - 1:
        logger.print(
            f"\nTraining stopped early at epoch "
            f"{epoch + 1}/{max_epochs}"
        )
    else:
        logger.print(
            "\nTraining completed for all epochs"
        )

    logger.print("\nSaving final model...")
    torch.save(
        {
            "feature_extractor": feature_extractor,
            "svgp": svgp,
            "likelihood": likelihood,
        },
        "checkpoints/model.pt",
    )
    wandb.finish()
    logger.print("✓ Model saved to model.pt!")
    logger.print("\n✓ Training completed successfully!")
    logger.print(
        f"Best validation metrics:\n"
        f"• Loss: {best_val_loss:.4f}\n"
        f"• R²: {best_val_r2:.4f}"
    )
    
    temp_model = torch.load(best_model_path)
    feature_extractor = temp_model["feature_extractor"]
    svgp = temp_model["svgp"]
    likelihood = temp_model["likelihood"]
    
    return feature_extractor, svgp, likelihood, train_dataloader, val_dataloader, test_dataloader