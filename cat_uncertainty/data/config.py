"""This module defines configurations using dataclasses."""

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from ..utils.config_utils import path_to_str


@dataclass
class DBParams:
    """Configuration for a single LMDB database."""

    db_path: str = (
        "/home/osman/Osman_Victus_HP/Shuwen_Group/UQ_OC20/UQ_Catalysis/"
        "uq_catalysis/data/catalysis_hub"
    )
    lmdb_files: str | Sequence[str] | None = "CatHub.lmdb"
    train_fraction: float = 0.7
    val_fraction: float = 0.1
    test_fraction: float = 0.2

    def __post_init__(self) -> None:
        """Validate field values."""
        if not 0.0 <= self.train_fraction <= 1.0:
            raise ValueError("train_fraction must be between 0.0 and 1.0")
        if not 0.0 <= self.val_fraction <= 1.0:
            raise ValueError("val_fraction must be between 0.0 and 1.0")
        if not 0.0 <= self.test_fraction <= 1.0:
            raise ValueError("test_fraction must be between 0.0 and 1.0")

        total = (
            self.train_fraction
            + self.val_fraction
            + self.test_fraction
        )
        if abs(total - 1.0) > 1e-6:  # Using small epsilon for float comparison
            raise ValueError(f"Fractions must sum to 1.0. Got {total}")

@dataclass
class SOAPTransformParams:
    """Configuration for SOAP transform parameters."""

    rcut: float = 5.0
    nmax: int = 3
    lmax: int = 3
    sigma: float = 0.5
    periodic: bool = True
    sparse: bool = False
    average: Literal["off", "inner", "outer"] = "off"

    def __post_init__(self) -> None:
        """Validate field values."""
        if self.rcut <= 0:
            raise ValueError("rcut must be positive")
        if self.nmax <= 0:
            raise ValueError("nmax must be positive")
        if self.lmax <= 0:
            raise ValueError("lmax must be positive")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        if self.average not in ["off", "inner", "outer"]:
            raise ValueError("average must be one of: off, inner, outer")


@dataclass
class Transform:
    """Configuration for data transformation."""

    name: str = "soap"
    params: SOAPTransformParams = field(default_factory=SOAPTransformParams)


@dataclass
class DataLoaderParams:
    """Configuration for dataloader/pl.LightningDataModule parameters."""

    batch_size: int = 64
    num_workers: int = 4
    train_shuffle: bool = True
    val_shuffle: bool = False
    test_shuffle: bool = False
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int = 2
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate field values."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")


@dataclass
class DataModuleConfig:
    """Configuration for train/val/test datasets and dataloader parameters."""

    data_params: DBParams
    transform: Transform | None = None
    dataloader_params: DataLoaderParams = field(
        default_factory=DataLoaderParams
    )

    def to_dict(self) -> dict:
        """Convert to dictionary with Path objects as strings."""
        return path_to_str(asdict(self))

    @classmethod
    def from_dict(cls, data: dict) -> "DataModuleConfig":
        """Create instance from dictionary."""
        if "db_type" not in data.get("data_params", {}):
            raise ValueError("data_params must contain db_type field")

        data["data_params"] = DBParams(**data["data_params"])

        if "transform" in data and data["transform"] is not None:
            transform_data = data["transform"]
            if "params" in transform_data:
                transform_data["params"] = SOAPTransformParams(
                    **transform_data["params"]
                )
            data["transform"] = Transform(**transform_data)

        if "dataloader_params" in data:
            data["dataloader_params"] = DataLoaderParams(
                **data["dataloader_params"]
            )

        return cls(**data)

    def to_json(self, file_path: str | Path) -> None:
        """Save to JSON file."""
        file_path = Path(file_path)
        file_path.write_text(json.dumps(self.to_dict(), indent=4))

    @classmethod
    def from_json(cls, file_path: str | Path) -> "DataModuleConfig":
        """Create instance from JSON file."""
        file_path = Path(file_path)
        return cls.from_dict(json.loads(file_path.read_text()))
