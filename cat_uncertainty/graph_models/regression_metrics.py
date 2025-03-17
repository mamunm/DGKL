"""Regression metrics for graph models."""


import torch
from torchmetrics import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
    R2Score,
)


class RegressionMetrics(torch.nn.Module):
    """A class to handle multiple regression metrics using torchmetrics.

    This class provides a unified interface for computing multiple regression
    metrics at once using torchmetrics, which is optimized for PyTorch and
    works well with PyTorch Lightning.

    Args:
        metrics (List[str], optional): List of metrics to compute. Defaults to
            ['mae', 'rmse', 'r2'].
            Available metrics: 'mae', 'rmse', 'r2', 'mse', 'mape'.
        stages (List[str], optional): List of stages to track metrics for.
            Defaults to ['training', 'validation', 'testing'].
        device (str, optional): Device to store the metrics on.
            Defaults to 'cpu'.

    Example:
        >>> metrics = RegressionMetrics(['mae', 'rmse', 'r2'])
        >>> # During training
        >>> metrics.update(pred, true, stage='training')
        >>> # During validation
        >>> metrics.update(pred, true, stage='validation')
        >>> # Get results
        >>> train_results = metrics.compute('training')
        >>> val_results = metrics.compute('validation')
    """

    AVAILABLE_METRICS = {
        "mae": "Mean Absolute Error",
        "rmse": "Root Mean Squared Error",
        "r2": "R² Score",
        "mse": "Mean Squared Error",
        "mape": "Mean Absolute Percentage Error",
    }

    STAGE_MAPPING = {
        "train": "training",
        "val": "validation",
        "test": "testing",
    }

    AVAILABLE_STAGES = ["training", "validation", "testing"]

    def __init__(
        self,
        metrics: list[str] | None = None,
        stages: list[str] | None = None,
    ):
        super().__init__()
        if metrics is None:
            metrics = ["mae", "rmse", "r2"]
        if stages is None:
            stages = ["training", "validation", "testing"]
        else:
            stages = [self.STAGE_MAPPING.get(s, s) for s in stages]

        invalid_metrics = set(metrics) - set(self.AVAILABLE_METRICS.keys())
        if invalid_metrics:
            raise ValueError(
                f"Invalid metrics: {invalid_metrics}. "
                f"Available metrics are: {list(self.AVAILABLE_METRICS.keys())}"
            )

        invalid_stages = set(stages) - set(self.AVAILABLE_STAGES)
        if invalid_stages:
            raise ValueError(
                f"Invalid stages: {invalid_stages}. "
                f"Available stages are: {self.AVAILABLE_STAGES}"
            )

        self.metrics = metrics
        self.stages = stages
        self._initialize_metrics()

    def _initialize_metrics(self) -> None:
        """Initialize all requested torchmetrics for each stage."""
        self.metric_computers = {}

        for stage in self.stages:
            stage_metrics = {}

            if "mae" in self.metrics:
                stage_metrics["mae"] = MeanAbsoluteError()
            if "mse" in self.metrics:
                stage_metrics["mse"] = MeanSquaredError()
            if "rmse" in self.metrics:
                stage_metrics["rmse"] = MeanSquaredError(squared=False)
            if "r2" in self.metrics:
                stage_metrics["r2"] = R2Score()
            if "mape" in self.metrics:
                stage_metrics["mape"] = MeanAbsolutePercentageError()

            self.metric_computers[stage] = stage_metrics

    def reset(self, stage: str | None = None) -> None:
        """Reset accumulated statistics for specified stage or all stages.

        Args:
            stage (str, optional): Stage to reset metrics for.
                If None, resets all stages. Can be 'train', 'val', 'test'
                or 'training', 'validation', 'testing'.
        """
        if stage is not None:
            stage = self.STAGE_MAPPING.get(stage, stage)
            if stage not in self.stages:
                raise ValueError(
                    f"Invalid stage: {stage}. Available stages: {self.stages}"
                )
            for metric in self.metric_computers[stage].values():
                metric.reset()
        else:
            for stage_metrics in self.metric_computers.values():
                for metric in stage_metrics.values():
                    metric.reset()

    def update(
        self, preds: torch.Tensor, targets: torch.Tensor, stage: str
    ) -> None:
        """
        Update the metrics with new preds and targets for a specific stage.

        Args:
            preds: Predicted values (torch.Tensor)
            targets: True values (torch.Tensor)
            stage: Stage to update metrics
        """
        stage = self.STAGE_MAPPING.get(stage, stage)
        if stage not in self.stages:
            raise ValueError(
                f"Invalid stage: {stage}. Available stages: {self.stages}"
            )

        for metric in self.metric_computers[stage].values():
            if metric.device != preds.device:
                metric.to(preds.device)
            metric.update(preds, targets)

    def compute(self, stage: str) -> dict[str, float]:
        """Compute all requested metrics for a specific stage.

        Args:
            stage: Stage to compute metrics for

        Returns:
            Dict[str, float]: Dictionary mapping metric names to their values
        """
        stage = self.STAGE_MAPPING.get(stage, stage)
        if stage not in self.stages:
            raise ValueError(
                f"Invalid stage: {stage}. Available stages: {self.stages}"
            )

        results = {}
        for name, metric in self.metric_computers[stage].items():
            results[name] = float(metric.compute().cpu())

        return results

    def compute_all(self) -> dict[str, dict[str, float]]:
        """Compute metrics for all stages.

        Returns:
            Dict[str, Dict[str, float]]: Dictionary mapping stages to
                their metric results
        """
        return {stage: self.compute(stage) for stage in self.stages}

    def __str__(self) -> str:
        """Return a string representation of the metrics configuration."""
        return (
            f"RegressionMetrics(metrics={self.metrics}, stages={self.stages})"
        )

    def __repr__(self) -> str:
        """Return a string representation of the metrics configuration."""
        return self.__str__()
