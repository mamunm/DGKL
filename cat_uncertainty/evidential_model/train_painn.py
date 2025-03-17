import os
import torch

from .evidential_trainer import EvidentialLightningModule, Optimizer
from .evidential import EvidentialModel
from ..graph_models.callbacks import get_callbacks_logger
import cat_uncertainty
from ..data.config import (
    DataLoaderParams,
    DataModuleConfig,
    DBParams,
)
from ..data.datamodule import GraphDataModule
from typing import Dict, Any, Literal

from cat_uncertainty.utils.console import Console
from ..graph_models.models.painn import PaiNN, PaiNNModelParams
import pytorch_lightning as pl
from pathlib import Path

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

def train_painn(
    config: Dict[str, Any],
    data: Literal["cathub", "eagle"] = "cathub",
    logger: Console | None = None,
    max_epochs: int = 200,
    checkpoint_dir: str = "checkpoints",
    seed: int = 42,
    project: str = "test",
    name: str = "test",
    htune: bool = False,
) -> EvidentialLightningModule:
    """Train a PaiNN model."""
    #######################Data Setup####################################################
    DB_PATH = (
        Path(cat_uncertainty.__path__[0]).parent
        / f"data/{DATA_DICT[data]}"
    )
    if logger:
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
    if logger:
        logger.print("Setting up data module...")
    data_module.setup()
    if logger:
        logger.print("Data module setup complete!")

    painn_params = PaiNNModelParams(
        num_layers=config.get("num_layers", 4),
        hidden_channels=config.get("hidden_channels", 128),
        num_radial=config.get("num_radial", 50),
        cutoff=config.get("cutoff", 10.0),
        max_neighbors=config.get("max_neighbors", 32),
        n_atom_type=config.get("n_atom_type", 100),
        output_dim=config.get("output_dim", 4),
        activation=config.get("activation", "ssp"),
        readout=config.get("readout", "mean"),
        dropout=config.get("dropout", None),
    )
    painn = PaiNN(config=painn_params)
    evidential_model = EvidentialModel(
        graph_model=painn,
        min_evidence=1e-5,
    )
    if logger:
        logger.print("PaiNN model created!")

    optimizer = Optimizer(
        optimizer=config.get("optimizer", "adamw"),
        scheduler=config.get("scheduler", "onecycle"),
        learning_rate=config.get("learning_rate", 1e-4),
        max_epochs=max_epochs,
        total_steps=max_epochs * len(data_module.train_dataloader()),
    )

    lightning_module = EvidentialLightningModule(
        model=evidential_model,
        optimizer=optimizer,
        max_epochs=max_epochs,
    )
    if logger:
        logger.print("Lightning module created!")

    callbacks, wandb_logger = get_callbacks_logger(
        dirpath=checkpoint_dir,
        logger=logger,
        project=project,
        name=name,
        htune=htune,
    )
    accumulate_grad_batches = {32: 1, 64: 1, 128: 1, 256: 2, 512: 4}

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=wandb_logger,
        log_every_n_steps=1,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=accumulate_grad_batches[config.get("batch_size", 64)],
    )

    if logger:
        logger.print("Training started...")

    trainer.fit(model=lightning_module, datamodule=data_module)

    if logger:
        logger.print("Training completed!")

    best_model = EvidentialLightningModule.load_from_checkpoint(callbacks[0].best_model_path)
    
    if not htune:
        return data_module, best_model