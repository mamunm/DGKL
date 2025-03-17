"""Module for analyzing graph model metrics and generating visualizations."""

import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class GraphMetricAnalysis:
    """Class for analyzing graph model metrics and generating visualizations."""

    def __init__(
        self,
        analysis_dir: Path,
        analysis_data: Dict[str, Dict[str, torch.Tensor]],
    ) -> None:
        """Initialize GraphMetricAnalysis.

        Args:
            analysis_dir: Directory to save analysis results
            analysis_data: Dictionary mapping splits to their analysis data containing:
                - y_true: True values
                - y_pred: Predicted values
                - y_pred_unc: Prediction uncertainties (if UQ task)
        """
        self.analysis_dir = analysis_dir
        self.analysis_data = analysis_data
        self.metrics: Dict[str, Dict[str, float]] = {}
        self._compute_metrics()

    def _compute_metrics(self) -> None:
        """Compute MAE, RMSE, and R² metrics for each split."""
        for split, data in self.analysis_data.items():
            y_true = data["y_true"].cpu().numpy()
            y_pred = data["y_pred"].cpu().numpy()

            self.metrics[split] = {
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "r2": float(r2_score(y_true, y_pred)),
            }

    def show_metric_analysis(self) -> None:
        """Display metrics in a rich table format."""
        console = Console()
        table = Table(title="Model Performance Metrics")

        table.add_column("Split", style="cyan")
        table.add_column("MAE", style="magenta")
        table.add_column("RMSE", style="green")
        table.add_column("R²", style="yellow")

        for split, metrics in self.metrics.items():
            table.add_row(
                split,
                f"{metrics['mae']:.4f}",
                f"{metrics['rmse']:.4f}",
                f"{metrics['r2']:.4f}",
            )

        console.print(table)

    def save_metric_analysis(self) -> None:
        """Save metrics to a JSON file.

        Args:
            output_dir: Directory to save the metrics file
        """
        metrics_file = self.analysis_dir / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(self.metrics, f, indent=4)

    def plot_metric_analysis(self) -> None:
        """Generate and save parity plots for each split."""
        plt.rcParams.update({"font.size": 12})  # Set global font size

        for split, data in self.analysis_data.items():
            y_true = data["y_true"].cpu().numpy()
            y_pred = data["y_pred"].cpu().numpy()

            # Create figure with gridspec for distributions
            fig = plt.figure(figsize=(10, 10))
            gs = fig.add_gridspec(3, 3)
            ax_main = fig.add_subplot(gs[1:, :-1])
            ax_top = fig.add_subplot(gs[0, :-1])
            ax_right = fig.add_subplot(gs[1:, -1])

            # Main scatter plot
            ax_main.plot(
                [
                    min(y_true.min(), y_pred.min()),
                    max(y_true.max(), y_pred.max()),
                ],
                [
                    min(y_true.min(), y_pred.min()),
                    max(y_true.max(), y_pred.max()),
                ],
                "k--",
                label="Parity",
                linewidth=2,
            )
            ax_main.scatter(y_true, y_pred, alpha=0.5, label="Predictions")
            ax_main.set_xlabel("Adsorption Energy [eV]", fontsize=12)
            ax_main.set_ylabel("Predicted Energy [eV]", fontsize=12)
            ax_main.set_title(
                f"Parity Plot - {split.capitalize()} Set", fontsize=12, pad=10
            )
            ax_main.legend(fontsize=12)

            # Add distributions
            ax_top.hist(y_true, bins=30, alpha=0.5, label="True", density=True)
            ax_top.hist(y_pred, bins=30, alpha=0.5, label="Pred", density=True)
            ax_top.legend(fontsize=12)
            ax_top.set_xticklabels([])

            ax_right.hist(
                y_pred,
                bins=30,
                alpha=0.5,
                orientation="horizontal",
                density=True,
            )
            ax_right.hist(
                y_true,
                bins=30,
                alpha=0.5,
                orientation="horizontal",
                density=True,
            )
            ax_right.set_yticklabels([])

            # Add metrics text
            metrics_text = (
                f"MAE: {self.metrics[split]['mae']:.4f}\n"
                f"RMSE: {self.metrics[split]['rmse']:.4f}\n"
                f"R²: {self.metrics[split]['r2']:.4f}"
            )
            ax_main.text(
                0.05,
                0.95,
                metrics_text,
                transform=ax_main.transAxes,
                verticalalignment="top",
                fontsize=12,
                bbox=dict(
                    boxstyle="round",
                    facecolor="white",
                    alpha=0.8,
                    edgecolor="black",
                    linewidth=2,
                ),
            )

            # Set thick borders
            for ax in [ax_main, ax_top, ax_right]:
                ax.spines["top"].set_linewidth(2)
                ax.spines["right"].set_linewidth(2)
                ax.spines["bottom"].set_linewidth(2)
                ax.spines["left"].set_linewidth(2)
                ax.tick_params(width=2)

            plt.tight_layout()
            plt.savefig(
                self.analysis_dir / f"parity_plot_{split}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()


if __name__ == "__main__":
    from pathlib import Path

    import torch

    torch.manual_seed(0)
    n_samples = 50

    x = torch.randn(n_samples, 3)

    data = {
        "train": {
            "y_true": x[:, 0],
            "y_pred": x[:, 0] + 0.4 * torch.randn(n_samples),
        },
        "valid": {
            "y_true": x[:, 1],
            "y_pred": x[:, 1] + 0.5 * torch.randn(n_samples),
        },
        "test": {
            "y_true": x[:, 2],
            "y_pred": x[:, 2] + 0.6 * torch.randn(n_samples),
        },
    }

    analysis = GraphMetricAnalysis(
        analysis_dir=Path("example_results"), analysis_data=data
    )

    analysis.show_metric_analysis()
    analysis.save_metric_analysis()
    analysis.plot_metric_analysis()
