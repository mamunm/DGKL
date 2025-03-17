"""Utility functions for dataclass configurations."""
from pathlib import Path
from typing import Any


def path_to_str(obj: Any) -> Any:
    """Convert Path objects to strings in nested structures.

    Args:
        obj: Any Python object that might contain Path objects

    Returns:
        The same object with all Path objects converted to strings
    """
    if isinstance(obj, dict):
        return {k: path_to_str(v) for k, v in obj.items()}
    if isinstance(obj, Path):
        return str(obj)
    return obj
