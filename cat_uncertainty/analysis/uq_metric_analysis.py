"""Module for analyzing uncertainty quantification metrics."""

import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from scipy import stats
from sklearn.metrics import (
    auc,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_curve,
)


class UQMetricAnalysis:
    """Class for analyzing uncertainty quantification metrics."""

    def __init__(
        self,
        analysis_dir: Path,
        analysis_data: Dict[str, Dict[str, torch.Tensor]],
    ) -> None:
        """Initialize UQMetricAnalysis.

        Args:
            analysis_dir: Directory to save analysis results
            analysis_data: Dictionary mapping splits to their analysis data containing:
                - y_true: True values
                - y_pred: Predicted values
                - y_pred_unc: Prediction uncertainties
        """
        self.analysis_dir = analysis_dir
        self.analysis_data = analysis_data
        self.regression_metrics: Dict[str, Dict[str, float]] = {}
        self.uncertainty_metrics: Dict[str, Dict[str, float]] = {}
        self._compute_metrics()

    def _compute_metrics(self) -> None:
        """Compute regression and uncertainty metrics for each split."""
        for split, data in self.analysis_data.items():
            y_true = data["y_true"].cpu().numpy()
            y_pred = data["y_pred"].cpu().numpy()
            y_pred_unc = data["y_pred_unc"].cpu().numpy()

            # Regression metrics
            self.regression_metrics[split] = {
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "r2": float(r2_score(y_true, y_pred)),
            }

            # Uncertainty metrics
            errors = np.abs(y_true - y_pred)
            picp_68 = self._compute_picp(
                y_true, y_pred, y_pred_unc, confidence=0.68
            )
            picp_95 = self._compute_picp(
                y_true, y_pred, y_pred_unc, confidence=0.95
            )
            miw_68 = self._compute_miw(y_pred_unc, confidence=0.68)
            miw_95 = self._compute_miw(y_pred_unc, confidence=0.95)
            nll = self._compute_nll(y_true, y_pred, y_pred_unc)

            # Compute binned metrics for RMSE and RMV
            n_bins = 10
            binned_metrics = self._compute_binned_metrics(
                y_true, y_pred, y_pred_unc, n_bins
            )

            # Compute calibration metrics
            calibration_metrics = self._compute_calibration_metrics(
                y_true,
                y_pred,
                y_pred_unc,
                binned_metrics["rmse"],
                binned_metrics["rmv"],
                n_bins,
            )

            self.uncertainty_metrics[split] = {
                "picp_68": picp_68,
                "picp_95": picp_95,
                "miw_68": miw_68,
                "miw_95": miw_95,
                "nll": nll,
                "ence": calibration_metrics["ence"],
                "ece": calibration_metrics["ece"],
                "miscalibration": float(np.mean(np.abs(errors - y_pred_unc))),
                "spearman_corr": float(stats.spearmanr(errors, y_pred_unc)[0]),
            }

    def _compute_nll(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_unc: np.ndarray,
    ) -> float:
        """Compute Negative Log-Likelihood for Gaussian distribution.

        Args:
            y_true: True values
            y_pred: Predicted mean values
            y_pred_unc: Predicted standard deviations

        Returns:
            Negative log-likelihood value
        """
        # Add small epsilon to avoid division by zero
        eps = 1e-8
        variance = y_pred_unc**2 + eps

        # Compute NLL components
        log_variance = np.log(variance)
        squared_error = (y_true - y_pred) ** 2

        # NLL formula for Gaussian: 0.5 * (log(2π) + log(σ²) + (y-μ)²/σ²)
        nll = 0.5 * (
            np.log(2 * np.pi) + log_variance + squared_error / variance
        )

        return float(np.mean(nll))

    def _compute_picp(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_unc: np.ndarray,
        confidence: float = 0.68,
    ) -> float:
        """Compute Prediction Interval Coverage Probability."""
        z_score = stats.norm.ppf((1 + confidence) / 2)
        lower = y_pred - z_score * y_pred_unc
        upper = y_pred + z_score * y_pred_unc
        coverage = np.mean((y_true >= lower) & (y_true <= upper))
        return float(coverage)

    def _compute_miw(
        self,
        y_pred_unc: np.ndarray,
        confidence: float = 0.68,
    ) -> float:
        """Compute Mean Interval Width."""
        z_score = stats.norm.ppf((1 + confidence) / 2)
        interval_width = 2 * z_score * y_pred_unc
        return float(np.mean(interval_width))

    def _compute_binned_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_unc: np.ndarray,
        n_bins: int = 10,
    ) -> Dict[str, np.ndarray]:
        """Compute RMSE and RMV metrics for binned uncertainties.

        Args:
            y_true: True values
            y_pred: Predicted values
            y_pred_unc: Prediction uncertainties
            n_bins: Number of bins for uncertainty values

        Returns:
            Dictionary containing binned metrics
        """
        # Sort data by uncertainty
        sort_idx = np.argsort(y_pred_unc)
        y_true = y_true[sort_idx]
        y_pred = y_pred[sort_idx]
        y_pred_unc = y_pred_unc[sort_idx]

        # Create bins
        bin_edges = np.array_split(np.arange(len(y_pred_unc)), n_bins)
        rmse_bins = []
        rmv_bins = []
        mean_unc_bins = []

        for bin_idx in bin_edges:
            if len(bin_idx) == 0:
                continue

            # Compute RMSE for bin
            rmse = np.sqrt(mean_squared_error(y_true[bin_idx], y_pred[bin_idx]))
            rmse_bins.append(rmse)

            # Compute RMV (Root of Mean Variance) for bin
            rmv = np.sqrt(np.mean(y_pred_unc[bin_idx] ** 2))
            rmv_bins.append(rmv)

            # Store mean uncertainty for bin center
            mean_unc_bins.append(np.mean(y_pred_unc[bin_idx]))

        metrics = {
            "rmse": np.array(rmse_bins),
            "rmv": np.array(rmv_bins),
            "mean_unc": np.array(mean_unc_bins),
        }

        return metrics

    def _compute_calibration_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_unc: np.ndarray,
        rmse: np.ndarray,
        rmv: np.ndarray,
        n_bins: int = 10,
    ) -> Dict[str, float]:
        """Compute ENCE and ECE metrics.

        Args:
            y_true: True values
            y_pred: Predicted values
            y_pred_unc: Predicted uncertainties
            rmse: Root Mean Square Error per bin
            rmv: Root Mean Variance per bin
            n_bins: Number of bins used

        Returns:
            Dictionary containing ENCE and ECE values
        """
        # Compute ENCE (Expected Normalized Calibration Error)
        ence = np.mean(np.abs(rmv - rmse) / rmv)

        # Compute ECE using confidence intervals
        z_values = np.linspace(0, 1, n_bins + 1)[1:]  # Exclude 0
        errors = np.abs(y_true - y_pred)

        ece = 0.0
        for z in z_values:
            # Compute confidence interval
            ci = z  # This is the expected coverage

            # Compute empirical frequency (proportion within the interval)
            pred_interval = z * y_pred_unc
            ef = np.mean((errors <= pred_interval).astype(float))

            # Add to ECE
            ece += np.abs(ci - ef)

        # Average over all bins
        ece /= n_bins

        return {
            "ence": float(ence),
            "ece": float(ece),
        }

    def _compute_auroc(
        self,
        errors: np.ndarray,
        uncertainties: np.ndarray,
        threshold: float = 0.5,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Compute AUROC for binary classification of errors based on uncertainty.

        Args:
            errors: Absolute prediction errors
            uncertainties: Predicted uncertainties
            threshold: Error threshold for binary classification

        Returns:
            AUROC score, false positive rates, true positive rates
        """
        # Convert errors to binary labels (1 if error > threshold)
        labels = (errors > threshold).astype(int)

        # Compute ROC curve
        fpr, tpr, _ = roc_curve(labels, uncertainties)
        auroc_score = auc(fpr, tpr)

        return auroc_score, fpr, tpr

    def show_metric_analysis(self) -> None:
        """Display metrics in rich table format."""
        console = Console()

        # Regression metrics table
        reg_table = Table(title="Model Performance Metrics (Regression)")
        reg_table.add_column("Split", style="cyan")
        reg_table.add_column("MAE", style="magenta")
        reg_table.add_column("RMSE", style="green")
        reg_table.add_column("R²", style="yellow")

        for split, metrics in self.regression_metrics.items():
            reg_table.add_row(
                split,
                f"{metrics['mae']:.4f}",
                f"{metrics['rmse']:.4f}",
                f"{metrics['r2']:.4f}",
            )

        # Uncertainty metrics table
        uq_table = Table(title="Model Performance Metrics (Uncertainty)")
        uq_table.add_column("Split", style="cyan")
        uq_table.add_column("PICP (68%)", style="magenta")
        uq_table.add_column("PICP (95%)", style="green")
        uq_table.add_column("MIW (68%)", style="yellow")
        uq_table.add_column("MIW (95%)", style="blue")
        uq_table.add_column("NLL", style="red")
        uq_table.add_column("ENCE", style="purple")
        uq_table.add_column("ECE", style="white")
        uq_table.add_column("Miscal.", style="red")
        uq_table.add_column("Err-Unc ρ", style="blue")

        for split, metrics in self.uncertainty_metrics.items():
            uq_table.add_row(
                split,
                f"{metrics['picp_68']:.4f}",
                f"{metrics['picp_95']:.4f}",
                f"{metrics['miw_68']:.4f}",
                f"{metrics['miw_95']:.4f}",
                f"{metrics['nll']:.4f}",
                f"{metrics['ence']:.4f}",
                f"{metrics['ece']:.4f}",
                f"{metrics['miscalibration']:.4f}",
                f"{metrics['spearman_corr']:.4f}",
            )

        console.print(reg_table)
        console.print("\n")
        console.print(uq_table)

    def save_metric_analysis(self) -> None:
        """Save metrics to JSON files."""
        metrics = {
            "regression": self.regression_metrics,
            "uncertainty": self.uncertainty_metrics,
        }

        metrics_file = self.analysis_dir / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)

    def plot_metric_analysis(self) -> None:
        """Generate and save UQ analysis plots."""
        plt.rcParams.update({"font.size": 12})

        for split, data in self.analysis_data.items():
            y_true = data["y_true"].cpu().numpy()
            y_pred = data["y_pred"].cpu().numpy()
            y_pred_unc = data["y_pred_unc"].cpu().numpy()

            # Create subplots with 2x3 grid
            fig = plt.figure(figsize=(25, 20))
            gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

            # 1. Error vs. Uncertainty plot (top-left)
            ax1 = fig.add_subplot(gs[0, 0])
            errors = np.abs(y_true - y_pred)
            ax1.scatter(y_pred_unc, errors, alpha=0.5)
            ax1.plot(
                [0, max(y_pred_unc)], [0, max(y_pred_unc)], "k--", label="Ideal"
            )
            ax1.set_xlabel("Predicted Uncertainty")
            ax1.set_ylabel("Absolute Error")
            ax1.set_title("Error vs. Uncertainty")
            ax1.legend()

            # 2. Calibration plot (top-middle)
            ax2 = fig.add_subplot(gs[0, 1])
            percentiles = np.arange(0, 101, 5)
            obs_freq = []
            for p in percentiles:
                z_score = stats.norm.ppf((100 + p) / 200)  # Convert to z-score
                pred_ints = z_score * y_pred_unc
                obs_freq.append(np.mean(errors <= pred_ints))

            ax2.plot(percentiles / 100, obs_freq, "-o", label="Model")
            ax2.plot([0, 1], [0, 1], "k--", label="Ideal")
            ax2.set_xlabel("Expected Confidence Level")
            ax2.set_ylabel("Observed Confidence Level")
            ax2.set_title("Calibration Plot")
            ax2.legend()

            # 3. AUROC curve (top-right)
            ax3 = fig.add_subplot(gs[0, 2])
            threshold = 0.5
            auroc_score, fpr, tpr = self._compute_auroc(
                errors, y_pred_unc, threshold
            )

            ax3.plot(fpr, tpr, "-", label=f"AUROC = {auroc_score:.4f}")
            ax3.plot([0, 1], [0, 1], "k--", label="Random")
            ax3.set_xlabel("False Positive Rate")
            ax3.set_ylabel("True Positive Rate")
            ax3.set_title(f"ROC Curve (Error > {threshold:.2f})")

            # Add threshold text
            bbox_props = dict(
                boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8
            )
            ax3.text(
                0.05,
                0.95,
                f"AUROC: {auroc_score:.4f}",
                transform=ax3.transAxes,
                verticalalignment="top",
                fontsize=12,
                family="monospace",
                bbox=bbox_props,
            )
            ax3.legend()

            # 4. PICP vs. MPIW plot (bottom-left)
            ax4 = fig.add_subplot(gs[1, 0])
            confidences = np.linspace(0.01, 0.99, 20)
            picps = []
            mpiws = []
            for conf in confidences:
                picps.append(
                    self._compute_picp(y_true, y_pred, y_pred_unc, conf)
                )
                mpiws.append(self._compute_miw(y_pred_unc, conf))

            ax4.plot(mpiws, picps, "-o")
            ax4.set_xlabel("Mean Prediction Interval Width")
            ax4.set_ylabel("Coverage Probability")
            ax4.set_title("Coverage vs. Width")

            # 5. RMSE vs RMV plot with calibration metrics (bottom-middle)
            ax5 = fig.add_subplot(gs[1, 1])
            n_bins = 10
            binned_metrics = self._compute_binned_metrics(
                y_true, y_pred, y_pred_unc, n_bins
            )
            calibration_metrics = self._compute_calibration_metrics(
                y_true,
                y_pred,
                y_pred_unc,
                binned_metrics["rmse"],
                binned_metrics["rmv"],
                n_bins,
            )

            # Plot RMSE vs RMV
            ax5.scatter(
                binned_metrics["rmse"], binned_metrics["rmv"], alpha=0.7
            )
            max_val = max(
                np.max(binned_metrics["rmse"]), np.max(binned_metrics["rmv"])
            )
            ax5.plot([0, max_val], [0, max_val], "k--", label="Ideal")
            ax5.set_xlabel("RMSE")
            ax5.set_ylabel("RMV")
            ax5.set_title("RMSE vs RMV")
            ax5.legend()

            # Add metrics as text in the plot
            nll = self.uncertainty_metrics[split]["nll"]
            metrics_text = (
                f"ENCE: {calibration_metrics['ence']:.4f}\n"
                f"ECE:  {calibration_metrics['ece']:.4f}\n"
                f"NLL:  {nll:.4f}"
            )

            bbox_props = dict(
                boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8
            )
            ax5.text(
                0.05,
                0.95,
                metrics_text,
                transform=ax5.transAxes,
                verticalalignment="top",
                fontsize=12,
                family="monospace",
                bbox=bbox_props,
            )

            # Set thick borders for all plots
            for ax in [ax1, ax2, ax3, ax4, ax5]:
                ax.spines["top"].set_linewidth(2)
                ax.spines["right"].set_linewidth(2)
                ax.spines["bottom"].set_linewidth(2)
                ax.spines["left"].set_linewidth(2)
                ax.tick_params(width=2)
                ax.title.set_fontsize(14)
                ax.xaxis.label.set_fontsize(12)
                ax.yaxis.label.set_fontsize(12)

            plt.savefig(
                self.analysis_dir / f"uq_analysis_{split}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()


if __name__ == "__main__":
    from pathlib import Path

    import torch

    torch.manual_seed(0)
    n_samples = 50

    # Create synthetic data with heteroscedastic noise
    x = torch.randn(n_samples, 4)
    noise_scale = 0.1 + 0.2 * torch.abs(
        x
    )  # Uncertainty increases with magnitude

    data = {
        "train": {
            "y_true": x[:, 0],
            "y_pred": x[:, 0] + noise_scale[:, 0] * torch.randn(n_samples),
            "y_pred_unc": noise_scale[:, 0],
        },
        "valid": {
            "y_true": x[:, 1],
            "y_pred": x[:, 1] + noise_scale[:, 1] * torch.randn(n_samples),
            "y_pred_unc": noise_scale[:, 1],
        },
        "test": {
            "y_true": x[:, 2],
            "y_pred": x[:, 2] + noise_scale[:, 2] * torch.randn(n_samples),
            "y_pred_unc": noise_scale[:, 2],
        },
        "calib": {
            "y_true": x[:, 3],
            "y_pred": x[:, 3] + noise_scale[:, 3] * torch.randn(n_samples),
            "y_pred_unc": noise_scale[:, 3],
        },
    }

    results_dir = Path("example_results")
    results_dir.mkdir(parents=True, exist_ok=True)

    analysis = UQMetricAnalysis(analysis_dir=results_dir, analysis_data=data)

    analysis.show_metric_analysis()
    analysis.save_metric_analysis()
    analysis.plot_metric_analysis()

    from .calibration import CalibrateModel

    calibrate_model = CalibrateModel(
        method="adaptive_conformal", analysis_data=data
    )
    calibrated_data = calibrate_model.calibrate_model()

    analysis = UQMetricAnalysis(
        analysis_dir=results_dir, analysis_data=calibrated_data
    )
    analysis.show_metric_analysis()
    analysis.save_metric_analysis()
    analysis.plot_metric_analysis()
