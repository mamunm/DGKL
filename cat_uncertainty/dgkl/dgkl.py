"""Deep Graph Kernel Learning."""

from collections.abc import Callable
from typing import Literal

import gpytorch
import torch
import torch.nn as nn
from torch_geometric.data import Batch


def construct_kernel(
    kernel_type: str, inducing_points: torch.Tensor
) -> gpytorch.kernels.Kernel:
    """Construct a kernel based on the specified type and initialize it with
       inducing points.

    Args:
        kernel_type (str): Type of kernel to construct ('rbf', 'matern',
            'exponentiallinear', 'product', or 'specialmixturekernel')
        inducing_points (torch.Tensor): Tensor of inducing points

    Returns:
        gpytorch.kernels.Kernel: Initialized kernel wrapped in ScaleKernel
    """
    with torch.no_grad():
        dists = torch.pdist(inducing_points)
        lengthscale = dists.mean().item()

    if kernel_type.lower() == "rbf":
        base_kernel = gpytorch.kernels.RBFKernel()
        base_kernel.lengthscale = lengthscale * torch.ones_like(
            base_kernel.lengthscale
        )
    elif kernel_type.lower() == "matern":
        base_kernel = gpytorch.kernels.MaternKernel(nu=2.5)
        base_kernel.lengthscale = lengthscale * torch.ones_like(
            base_kernel.lengthscale
        )
    elif kernel_type.lower() == "exponentiallinear":
        base_kernel = gpytorch.kernels.ExponentialLinearKernel()
        base_kernel.lengthscale = lengthscale * torch.ones_like(
            base_kernel.lengthscale
        )
    elif kernel_type.lower() == "product":
        rbf_kernel = gpytorch.kernels.RBFKernel()
        rbf_kernel.lengthscale = lengthscale * torch.ones_like(
            rbf_kernel.lengthscale
        )
        exponential_kernel = gpytorch.kernels.ExponentialLinearKernel()
        base_kernel = gpytorch.kernels.ProductKernel(
            rbf_kernel, exponential_kernel
        )
    elif kernel_type.lower() == "specialmixturekernel":
        base_kernel = gpytorch.kernels.SpecialMixtureKernel()
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    return gpytorch.kernels.ScaleKernel(base_kernel)


def get_variational_distribution(
    dist_type: Literal["cholesky", "meanfield", "delta", "natural"],
    num_inducing_points: int,
) -> gpytorch.variational._VariationalDistribution:
    """Get the variational distribution based on the specified type.

    Args:
        dist_type (str): Type of variational distribution
            ('cholesky', 'meanfield', or 'delta')
        num_inducing_points (int): Number of inducing points

    Returns:
        gpytorch.variational.VariationalDistribution: The variational
            distribution
    """
    if dist_type.lower() == "cholesky":
        return gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points
        )
    elif dist_type.lower() == "meanfield":
        return gpytorch.variational.MeanFieldVariationalDistribution(
            num_inducing_points
        )
    elif dist_type.lower() == "delta":
        return gpytorch.variational.DeltaVariationalDistribution(
            num_inducing_points
        )
    elif dist_type.lower() == "natural":
        return gpytorch.variational.NaturalVariationalDistribution(
            num_inducing_points,
            mean_init_std=0.01
        )
    else:
        raise ValueError(
            f"Unknown variational distribution type: {dist_type}. "
            "Must be one of: 'cholesky', 'meanfield', 'delta', or 'natural'"
        )


class SVGP(gpytorch.models.ApproximateGP):
    """Stochastic Variational Gaussian Process.

    Args:
        inducing_points (torch.Tensor): Inducing points
        kernel_type (str, optional): Type of kernel to use. Defaults to "rbf".
        dist_type (str, optional): Type of variational distribution. Defaults
            to "cholesky".
        variational_strategy (str, optional): Type of variational strategy.
            Defaults to "standard".
    """

    def __init__(
        self,
        inducing_points: torch.Tensor,
        kernel_type: str = "rbf",
        dist_type: Literal["cholesky", "meanfield", "delta", "natural"] = "cholesky",
        variational_strategy: Literal["standard", "decoupled"] = "standard",
    ) -> None:
        if variational_strategy == "decoupled":
            num_covar_inducing = int(inducing_points.size(0) * 0.1)
            covar_inducing_points = inducing_points[
                torch.randperm(inducing_points.size(0))[:num_covar_inducing]
            ]
            covar_variational_strategy = (
                gpytorch.variational.VariationalStrategy(
                    self,
                    covar_inducing_points,
                    gpytorch.variational.CholeskyVariationalDistribution(
                        num_covar_inducing
                    ),
                    learn_inducing_locations=True,
                )
            )
            variational_strategy = (
                gpytorch.variational.OrthogonallyDecoupledVariationalStrategy(
                    covar_variational_strategy,
                    inducing_points,
                    gpytorch.variational.DeltaVariationalDistribution(
                        inducing_points.size(0)
                    ),
                )
            )
        else:
            variational_distribution = get_variational_distribution(
                dist_type=dist_type, num_inducing_points=inducing_points.size(0)
            )
            variational_strategy = gpytorch.variational.VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=True,
            )

        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = construct_kernel(kernel_type, inducing_points)

    def forward(
        self,
        x: torch.Tensor,
    ) -> gpytorch.distributions.MultivariateNormal:
        """Forward pass of the GP model.

        Args:
            x: features

        Returns:
            MultivariateNormal distribution
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def predict(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get predictions with uncertainty.

        Args:
            x: features

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing:
                - mean predictions [batch_size, output_dim]
                - standard deviation of predictions [batch_size, output_dim]
        """
        with torch.no_grad():
            f_dist = self(x)
            y_dist = self.likelihood(f_dist)

            mean = y_dist.mean
            std = y_dist.stddev

        return mean, std


class DGKL(nn.Module):
    """Deep Graph Kernel Learning model.

    Args:
        inducing_points (torch.Tensor): Inducing points
        feature_extractor (Optional[nn.Module | Callable[[torch.Tensor],
            torch.Tensor]]): Feature extractor
        kernel_type (str, optional): Type of kernel to use. Defaults to "rbf".
        dist_type (str, optional): Type of variational distribution.
            Defaults to "cholesky".
        variational_strategy (str, optional): Type of variational strategy.
            Defaults to "standard".
    """

    def __init__(
        self,
        inducing_points: torch.Tensor,
        feature_extractor: nn.Module
        | Callable[[torch.Tensor], torch.Tensor]
        | None = None,
        kernel_type: str = "rbf",
        dist_type: Literal["cholesky", "meanfield", "delta"] = "cholesky",
        variational_strategy: Literal["standard", "decoupled"] = "standard",
    ) -> None:
        super().__init__()

        if feature_extractor is not None:
            self.feature_extractor = feature_extractor
        else:
            self.feature_extractor = None

        self.gp = SVGP(
            inducing_points,
            kernel_type=kernel_type,
            dist_type=dist_type,
            variational_strategy=variational_strategy,
        )

    def forward(
        self,
        inputs: Batch
        | torch.Tensor
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> gpytorch.distributions.MultivariateNormal:
        """Forward pass of the GP model.

        Args:
            inputs: Tuple containing:
                - x: Node features tensor [N, D] or atomic numbers [N]
                - second: Edge index tensor [2, E] or positions [N, 3]
                - batch: Batch assignment tensor [N]

        Returns:
            MultivariateNormal distribution
        """
        if self.feature_extractor is not None:
            if isinstance(inputs, Batch):
                features = self.feature_extractor(inputs)
            else:
                features = self.feature_extractor(*inputs)
        else:
            features = inputs

        return self.gp(features)

    def predict(
        self,
        inputs: Batch
        | torch.Tensor
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get predictions with uncertainty.

        Args:
            inputs: Tuple containing:
                - x: Node features tensor [N, D] or atomic numbers [N]
                - second: Edge index tensor [2, E] or positions [N, 3]
                - batch: Batch assignment tensor [N]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing:
                - mean predictions [batch_size, output_dim]
                - standard deviation of predictions [batch_size, output_dim]
        """
        if self.feature_extractor is not None:
            if isinstance(inputs, Batch):
                features = self.feature_extractor(inputs)
            else:
                features = self.feature_extractor(*inputs)
        else:
            features = inputs

        return self.gp.predict(features)
