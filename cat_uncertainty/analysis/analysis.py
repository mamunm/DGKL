"""Analysis module for model evaluation and uncertainty quantification."""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
from torch.nn import Module

from ..dataclass.config import Config
from ..dataclass.model_config import FairchemFeatureExtractor
from ..utils import Logger
from .calibration import CalibrateModel
from .graph_metric_analysis import GraphMetricAnalysis
from .uq_metric_analysis import UQMetricAnalysis

CalibrateModelOptions = Literal[
    "platt",
    "platt_logit",
    "isotonic",
    "temperature",
    "conformal",
    "adaptive_conformal",
]


class ModelAnalyzer:
    """Generic analyzer for model evaluation and uncertainty quantification."""

    def __init__(
        self,
        config: Config,
        model: Module,
        data: Any,
        logger: Optional[Logger] = None,
        post_hoc_calibration: bool = False,
        calibration_model: Optional[CalibrateModelOptions] = None,
    ) -> None:
        """Initialize ModelAnalyzer.

        Args:
            config: Configuration object
            model: PyTorch model to analyze
            data: Data module containing datasets
            post_hoc_calibration: Whether to use post-hoc calibration
            calibration_model: Calibration model to use
        """
        if config.model.task_type == "graph" and post_hoc_calibration:
            raise ValueError(
                "Post-hoc calibration is not supported for graph training."
            )
        self.config = config
        self.dir = Path(self.config.experiment_dir) / "analysis"
        self.data = data
        self.data.setup()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.logger = logger
        self.post_hoc_calibration = post_hoc_calibration
        self.calibration_model = calibration_model

    def get_dataloader(self, split: str) -> Any:
        """Get dataloader for specified split.

        Args:
            split: Dataset split to get loader for ("train", "val", "test", or "calib")

        Returns:
            DataLoader for the specified split

        Raises:
            ValueError: If split is not one of "train", "val", "test", or "calib"
        """
        if split == "train":
            return self.data.train_dataloader()
        elif split == "val":
            return self.data.val_dataloader()
        elif split == "test":
            return self.data.test_dataloader()
        elif split == "calib":
            return self.data.cal_dataloader()
        else:
            raise ValueError(
                f'Split must be one of "train", "val", "test", or "calib", got {split}'
            )

    def _get_valid_splits(self, split: str) -> List[str]:
        """Get list of valid splits based on input split and configuration.

        Args:
            split: Dataset to analyze. Can be "train", "val", "test", "calib", or "all"

        Returns:
            List of valid splits to analyze

        Raises:
            ValueError: If split is not valid
        """
        valid_splits = ["train", "val", "test"]
        if self.post_hoc_calibration:
            valid_splits.append("calib")

        if split == "all":
            return valid_splits
        elif split in valid_splits:
            return [split]
        else:
            available_splits = ", ".join(f"'{s}'" for s in valid_splits)
            raise ValueError(
                f"Split must be one of {available_splits}, or 'all', got '{split}'"
            )

    def _process_graph_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process batch for graph regression task.

        Args:
            batch: Dictionary containing batch data

        Returns:
            Tuple of (predictions, true values)
        """
        output = self.model(
            batch["atomic_numbers"].long().to(self.device),
            batch["pos"].to(self.device),
            batch["batch"].to(self.device),
        )
        return output.view(-1), batch["energy"].to(self.device).view(-1)

    def _process_svgp_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process batch for SVGP model.

        Args:
            batch: Dictionary containing batch data

        Returns:
            Tuple of (predictions, uncertainties, true values)
        """
        if self.config.data.transform:
            features = (
                batch["x"].to(self.device),
                batch["edge_index"].to(self.device),
                batch["batch"].to(self.device),
            )
        elif isinstance(
            self.config.model.model_params.feature_extractor,
            FairchemFeatureExtractor,
        ):
            features = (batch.to(self.device),)
        else:
            features = (
                batch["atomic_numbers"].long().to(self.device),
                batch["pos"].to(self.device),
                batch["batch"].to(self.device),
            )

        mean, std = self.model.predict(features)
        return (
            mean.view(-1),
            std.view(-1),
            batch["energy"].to(self.device).view(-1),
        )

    def _process_derived_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process batch for derived (ensemble or mcd) model.

        Args:
            batch: Dictionary containing batch data

        Returns:
            Tuple of (predictions, uncertainties, true values)
        """

        mean, std = self.model(
            batch["atomic_numbers"].long().to(self.device),
            batch["pos"].to(self.device),
            batch["batch"].to(self.device),
        )

        return (
            mean.view(-1),
            std.view(-1),
            batch["energy"].to(self.device).view(-1),
        )

    def _process_evidential_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process batch for Evidential Deep Learning model.

        Args:
            batch: Dictionary containing batch data

        Returns:
            Tuple of (predictions, uncertainties, true values)
        """
        mu, lambda_, alpha, beta = self.model(
            batch["atomic_numbers"].long().to(self.device),
            batch["pos"].to(self.device),
            batch["batch"].to(self.device),
        )

        mean = mu
        var = beta * (1 + 1 / lambda_) / (alpha - 1)
        std = torch.sqrt(var)

        return (
            mean.view(-1),
            std.view(-1),
            batch["energy"].to(self.device).view(-1),
        )

    def _process_batches(self, dataloader: Any) -> Dict[str, torch.Tensor]:
        """Process batches from dataloader to collect predictions and true values.

        Args:
            dataloader: DataLoader containing batches to process

        Returns:
            Dictionary containing collected tensors:
                - y_true: True values
                - y_pred: Predicted values
                - y_pred_unc: Prediction uncertainties (if UQ task)
        """
        y_true_list = []
        y_pred_list = []
        y_pred_unc_list = []

        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                if self.config.model.task_type == "graph":
                    y_pred, y_true = self._process_graph_batch(batch)
                    y_pred_list.append(y_pred)
                    y_true_list.append(y_true)
                elif self.config.model.task_type == "uq":
                    uq_type = self.config.model.model_params.uq_type
                    if uq_type == "svgp":
                        y_pred, y_pred_unc, y_true = self._process_svgp_batch(
                            batch
                        )
                    elif uq_type in [
                        "ensemble",
                        "mcd",
                    ]:
                        (
                            y_pred,
                            y_pred_unc,
                            y_true,
                        ) = self._process_derived_batch(batch)
                    elif uq_type == "evidential":
                        (
                            y_pred,
                            y_pred_unc,
                            y_true,
                        ) = self._process_evidential_batch(batch)
                    else:
                        raise ValueError(
                            f"Unknown UQ model: {self.config.model.name}"
                        )
                    y_pred_list.append(y_pred)
                    y_pred_unc_list.append(y_pred_unc)
                    y_true_list.append(y_true)
                else:
                    raise ValueError(
                        f"Unknown task: {self.config.model.task_type}"
                    )

        result = {
            "y_true": torch.cat(y_true_list),
            "y_pred": torch.cat(y_pred_list),
        }

        if self.config.model.task_type == "uq_training":
            result["y_pred_unc"] = torch.cat(y_pred_unc_list)

        return result

    def _save_analysis_data(
        self, analysis_data: Dict[str, Dict[str, torch.Tensor]]
    ) -> None:
        """Save analysis data to disk.

        Args:
            analysis_data: Dictionary mapping splits to their analysis data containing:
                - y_true: True values
                - y_pred: Predicted values
                - y_pred_unc: Prediction uncertainties (if UQ task)
        """
        self.dir.mkdir(parents=True, exist_ok=True)

        self.logger.log_time_message("Saving analysis data.")
        for split, data in analysis_data.items():
            save_path = self.dir / f"{split}_predictions.pt"
            torch.save(data, save_path)

    def run_analysis(
        self, split: str = "all"
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Run analysis on specified dataset split(s).

        Args:
            split: Dataset to analyze. Can be "train", "val", "test", "calib", or "all"

        Returns:
            Dictionary mapping splits to their analysis data containing:
                - y_true: True values
                - y_pred: Predicted values
                - y_pred_unc: Prediction uncertainties (if UQ task)
        """
        splits = self._get_valid_splits(split)
        analysis_data = {}

        for current_split in splits:
            self.logger.log_time_message(
                f"Running analysis on {current_split} data."
            )
            dataloader = self.get_dataloader(current_split)
            analysis_data[current_split] = self._process_batches(dataloader)

        self._save_analysis_data(analysis_data)

        if self.config.model.task_type == "graph":
            metrics = GraphMetricAnalysis(self.dir, analysis_data)
        else:
            metrics = UQMetricAnalysis(self.dir, analysis_data)
        metrics.show_metric_analysis()
        metrics.save_metric_analysis()
        metrics.plot_metric_analysis()

        if self.post_hoc_calibration:
            calib_model = CalibrateModel(
                method=self.calibration_model, analysis_data=analysis_data
            )
            calibrated_analysis_data = calib_model.calibrate_model()

            calibrated_dir = self.dir / "calibrated"
            calibrated_dir.mkdir(parents=True, exist_ok=True)
            calibrated_metrics = UQMetricAnalysis(
                calibrated_dir, calibrated_analysis_data
            )
            calibrated_metrics.show_metric_analysis()
            calibrated_metrics.save_metric_analysis()
            calibrated_metrics.plot_metric_analysis()

        return analysis_data
