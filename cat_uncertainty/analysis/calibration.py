"""Module for post-hoc calibration of uncertainty estimates."""

from abc import ABC, abstractmethod
from typing import Dict, Literal

import numpy as np
import torch
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, LogisticRegression


class BaseCalibrator(ABC):
    """Base class for uncertainty calibration methods."""

    def __init__(self) -> None:
        """Initialize calibrator."""
        self.is_fitted = False

    @abstractmethod
    def fit(self, errors: np.ndarray, uncertainties: np.ndarray) -> None:
        """Fit calibration model.

        Args:
            errors: Absolute errors between true and predicted values
            uncertainties: Model's uncertainty predictions
        """
        pass

    @abstractmethod
    def predict(self, uncertainties: np.ndarray) -> np.ndarray:
        """Predict calibrated uncertainties.

        Args:
            uncertainties: Model's uncertainty predictions

        Returns:
            Calibrated uncertainties
        """
        pass


class PlattCalibrator(BaseCalibrator):
    """Platt scaling using linear regression."""

    def __init__(self) -> None:
        """Initialize Platt calibrator."""
        super().__init__()
        self.model = LinearRegression()

    def fit(self, errors: np.ndarray, uncertainties: np.ndarray) -> None:
        """Fit linear regression model."""
        x = uncertainties.reshape(-1, 1)
        self.model.fit(x, errors)
        self.is_fitted = True

    def predict(self, uncertainties: np.ndarray) -> np.ndarray:
        """Predict calibrated uncertainties."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self.model.predict(uncertainties.reshape(-1, 1))


class PlattLogitCalibrator(BaseCalibrator):
    """Platt scaling using logistic regression."""

    def __init__(self) -> None:
        """Initialize Platt-logit calibrator."""
        super().__init__()
        self.model = LogisticRegression(random_state=0)

    def fit(self, errors: np.ndarray, uncertainties: np.ndarray) -> None:
        """Fit logistic regression model."""
        x = uncertainties.reshape(-1, 1)
        y = (errors <= uncertainties).astype(int)
        self.model.fit(x, y)
        self.is_fitted = True

    def predict(self, uncertainties: np.ndarray) -> np.ndarray:
        """Predict calibrated uncertainties."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        x = uncertainties.reshape(-1, 1)
        return self.model.predict_proba(x)[:, 1] * uncertainties


class IsotonicCalibrator(BaseCalibrator):
    """Isotonic regression calibration."""

    def __init__(self) -> None:
        """Initialize isotonic calibrator."""
        super().__init__()
        self.model = IsotonicRegression(out_of_bounds="clip")

    def fit(self, errors: np.ndarray, uncertainties: np.ndarray) -> None:
        """Fit isotonic regression model."""
        self.model.fit(uncertainties, errors)
        self.is_fitted = True

    def predict(self, uncertainties: np.ndarray) -> np.ndarray:
        """Predict calibrated uncertainties."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self.model.predict(uncertainties)


class TemperatureCalibrator(BaseCalibrator):
    """Temperature scaling calibration."""

    def __init__(self) -> None:
        """Initialize temperature calibrator."""
        super().__init__()
        self.temperature = 1.0

    def _nll_loss(
        self, temp: float, errors: np.ndarray, uncertainties: np.ndarray
    ) -> float:
        """Compute negative log likelihood loss."""
        scaled_uncertainties = uncertainties * temp
        return float(
            np.mean(
                (errors - scaled_uncertainties) ** 2
                / (2 * scaled_uncertainties**2)
                + np.log(scaled_uncertainties)
            )
        )

    def fit(self, errors: np.ndarray, uncertainties: np.ndarray) -> None:
        """Find optimal temperature parameter."""
        result = minimize(
            lambda temp: self._nll_loss(temp, errors, uncertainties),
            x0=1.0,
            method="Nelder-Mead",
        )
        self.temperature = float(result.x[0])
        self.params = {"temperature": self.temperature}
        self.is_fitted = True

    def predict(self, uncertainties: np.ndarray) -> np.ndarray:
        """Predict calibrated uncertainties."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return uncertainties * self.temperature


class ConformalCalibrator(BaseCalibrator):
    """Conformal prediction calibration."""

    def __init__(self, confidence: float = 0.95) -> None:
        """Initialize conformal calibrator.

        Args:
            confidence: Desired confidence level
        """
        super().__init__()
        self.confidence = confidence
        self.factor = 1.0

    def fit(self, errors: np.ndarray, uncertainties: np.ndarray) -> None:
        """Find calibration factor."""
        scores = errors / uncertainties
        n = len(scores)
        level = np.ceil((n + 1) * self.confidence) / n
        self.factor = float(np.quantile(scores, level))
        self.params = {"factor": self.factor}
        self.is_fitted = True

    def predict(self, uncertainties: np.ndarray) -> np.ndarray:
        """Predict calibrated uncertainties."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return uncertainties * self.factor


class AdaptiveConformalCalibrator(BaseCalibrator):
    def __init__(self, confidence: float = 0.95, n_bins: int = 10) -> None:
        super().__init__()
        self.confidence = confidence
        self.n_bins = n_bins
        self.factors = None

    def fit(self, errors: np.ndarray, uncertainties: np.ndarray) -> None:
        # Bin the data based on uncertainty magnitude
        bins = np.percentile(
            uncertainties, np.linspace(0, 100, self.n_bins + 1)
        )
        self.factors = []

        for i in range(self.n_bins):
            mask = (uncertainties >= bins[i]) & (uncertainties <= bins[i + 1])
            if np.sum(mask) > 0:
                scores = errors[mask] / uncertainties[mask]
                n = len(scores)
                level = np.ceil((n + 1) * self.confidence) / n
                self.factors.append(float(np.quantile(scores, level)))
            else:
                self.factors.append(1.0)

        self.bins = bins
        self.is_fitted = True

    def predict(self, uncertainties: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        calibrated = np.zeros_like(uncertainties)
        for i in range(self.n_bins):
            mask = (uncertainties >= self.bins[i]) & (
                uncertainties <= self.bins[i + 1]
            )
            calibrated[mask] = uncertainties[mask] * self.factors[i]

        return calibrated


class CalibrateModel:
    """Class for post-hoc calibration of uncertainty estimates."""

    def __init__(
        self,
        method: Literal[
            "platt",
            "platt_logit",
            "isotonic",
            "temperature",
            "conformal",
            "adaptive_conformal",
        ],
        analysis_data: Dict[str, Dict[str, torch.Tensor]],
    ) -> None:
        """Initialize CalibrateModel.

        Args:
            method: Calibration method to use
            analysis_data: Dictionary mapping splits to their analysis data containing:
                - y_true: True values
                - y_pred: Predicted values
                - y_pred_unc: Prediction uncertainties
        """
        self.method = method
        self.analysis_data = analysis_data

        # Initialize appropriate calibrator
        if method == "platt":
            self.calibrator = PlattCalibrator()
        elif method == "platt_logit":
            self.calibrator = PlattLogitCalibrator()
        elif method == "isotonic":
            self.calibrator = IsotonicCalibrator()
        elif method == "temperature":
            self.calibrator = TemperatureCalibrator()
        elif method == "conformal":
            self.calibrator = ConformalCalibrator()
        elif method == "adaptive_conformal":
            self.calibrator = AdaptiveConformalCalibrator()
        else:
            raise ValueError(f"Unknown calibration method: {method}")

    def calibrate_model(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Calibrate model uncertainties using specified method.

        Returns:
            Dictionary with calibrated uncertainties for each split
        """
        # Get calibration data
        calib_data = self.analysis_data["calib"]
        y_true_calib = calib_data["y_true"].cpu().numpy()
        y_pred_calib = calib_data["y_pred"].cpu().numpy()
        y_pred_unc_calib = calib_data["y_pred_unc"].cpu().numpy()

        # Fit calibrator on calibration set
        errors_calib = np.abs(y_true_calib - y_pred_calib)
        self.calibrator.fit(errors_calib, y_pred_unc_calib)

        # Apply calibration to all splits
        calibrated_data = {}
        for split, data in self.analysis_data.items():
            y_pred_unc = data["y_pred_unc"].cpu().numpy()
            calibrated_unc = self.calibrator.predict(y_pred_unc)

            calibrated_data[split] = {
                "y_true": data["y_true"],
                "y_pred": data["y_pred"],
                "y_pred_unc": torch.tensor(
                    calibrated_unc, device=data["y_true"].device
                ),
            }

        return calibrated_data


if __name__ == "__main__":
    # Example usage
    import torch

    torch.manual_seed(0)
    n_samples = 100

    # Create synthetic data with miscalibrated uncertainties
    x = torch.randn(n_samples, 2)
    true_noise = 0.1 + 0.2 * torch.abs(x)  # True heteroscedastic noise
    pred_noise = 0.05 + 0.1 * torch.abs(x)  # Underestimated uncertainties

    data = {
        "calib": {
            "y_true": x[:, 0],
            "y_pred": x[:, 0] + true_noise[:, 0] * torch.randn(n_samples),
            "y_pred_unc": pred_noise[:, 0],
        },
        "test": {
            "y_true": x[:, 1],
            "y_pred": x[:, 1] + true_noise[:, 1] * torch.randn(n_samples),
            "y_pred_unc": pred_noise[:, 1],
        },
    }

    # Test all calibration methods
    for method in [
        "platt",
        "platt_logit",
        "isotonic",
        "temperature",
        "conformal",
    ]:
        print(f"\nTesting {method} calibration:")
        calibrator = CalibrateModel(method=method, analysis_data=data)
        calibrated_data = calibrator.calibrate_model()
