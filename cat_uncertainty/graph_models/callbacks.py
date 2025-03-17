from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from pytorch_lightning.callbacks import Callback
from typing import List
from rich.console import Console
import torch

custom_theme = RichProgressBarTheme(
    description="green_yellow",
    progress_bar="green1",
    progress_bar_finished="green1",
    progress_bar_pulse="#6206E0",
    batch_progress="yellow",
    time="grey82",
    processing_speed="grey82",
    metrics="grey82",
    metrics_text_delimiter="\n",
    metrics_format=".3e"  # Scientific notation for metrics
)

class NaNDetectorCallback(Callback):
    """
    Callback that detects and handles NaN values in the model.

    Args:
        logger: Console logger.
        lr_reduction_factor: Factor by which the learning rate is reduced if NaN is detected.
        min_lr: Minimum learning rate.
    """
    def __init__(self, logger: Console,lr_reduction_factor=0.5, min_lr=1e-8):
        super().__init__()
        self.logger = logger
        self.lr_reduction_factor = lr_reduction_factor
        self.min_lr = min_lr
        self.previous_weights = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """
        Save current model state.
        """
        self.previous_weights = {
            name: param.clone().detach()
            for name, param in pl_module.named_parameters()
        }

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """
        Detect and handle NaN values in the model.
        """
        if self._contains_nan(outputs) or self._model_contains_nan(pl_module):
            self._rewind_model_state(pl_module)
            self._reduce_learning_rate(trainer)
            
            # Skip rest of batch
            trainer.should_stop = True

    def _contains_nan(self, outputs):
        if outputs is None:
            return False
        loss = outputs['loss'] if isinstance(outputs, dict) else outputs
        return torch.isnan(loss).any()

    def _model_contains_nan(self, pl_module):
        return any(torch.isnan(param).any() for param in pl_module.parameters())

    def _rewind_model_state(self, pl_module):
        if self.previous_weights is not None:
            for name, param in pl_module.named_parameters():
                param.data.copy_(self.previous_weights[name])

    def _reduce_learning_rate(self, trainer):
        for param_group in trainer.optimizers[0].param_groups:
            current_lr = param_group['lr']
            new_lr = max(current_lr * self.lr_reduction_factor, self.min_lr)
            param_group['lr'] = new_lr
            if self.logger is not None:
                self.logger.print(f"Reducing learning rate from {current_lr} to {new_lr}")

def get_callbacks_logger(
    dirpath: str,
    logger: Console | None = None,
    project: str = "test",
    name: str = "test",
    htune: bool = False,
) -> List[Callback | WandbLogger]:
    """
    Returns a list of callbacks and a logger for use in a PyTorch Lightning LightningModule.

    Args:
        dirpath: The directory path to save the model checkpoints.
        project: The name of the project in Weights and Biases.
        name: The name of the run in Weights and Biases.
        htune: Whether to use Hyper Tune for training.

    Returns:
        A list of callbacks and a logger.
    """
    callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            dirpath=dirpath,
            mode="min",
            save_top_k=1,
            filename="{epoch:02d}-{val_loss:.4f}",
        ),
        # ModelCheckpoint(
        #     dirpath=dirpath,
        #     filename="model-{epoch:02d}",
        #     every_n_epochs=50,
        #     save_top_k=-1,
        # ),
        LearningRateMonitor(logging_interval="epoch"),
        RichModelSummary(max_depth=3),
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            min_delta=1e-5,
            mode="min",
        ),
        RichProgressBar(theme=custom_theme),
        NaNDetectorCallback(
            logger=logger,
            lr_reduction_factor=0.5,
            min_lr=1e-8,
        ),
    ]

    if htune:
        callbacks.append(
            TuneReportCallback(metrics="val_loss", on="validation_end")
        )

    logger = WandbLogger(
        project=project,
        name=name,
        save_dir=dirpath,
    )

    return callbacks, logger