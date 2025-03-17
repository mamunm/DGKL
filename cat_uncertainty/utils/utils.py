"""Utility functions for the cat_uncertainty package."""

import contextlib
import datetime
import inspect
import os
import sys
from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from rich.text import Text


@contextlib.contextmanager
def suppress_output():
    """Context manager to suppress stdout and stderr."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


class Logger:
    """Simple logger with timing functionality."""

    def __init__(self) -> None:
        """Initialize logger with current time."""
        self.start_time = datetime.datetime.now()
        self.console = Console()

    def log_time_message(self, message: str) -> None:
        """Log a message with the current time and file name.

        Args:
            message: Message to log
        """
        current_time = datetime.datetime.now()
        elapsed_time = current_time - self.start_time

        # Get the caller's frame information
        frame = inspect.currentframe()
        caller_frame = frame.f_back if frame is not None else None
        file_name = (
            Path(caller_frame.f_code.co_filename).stem
            if caller_frame is not None
            else ""
        )

        # Create formatted text with rich
        text = Text()
        text.append("[", style="white")
        text.append(
            current_time.strftime("%Y-%m-%d %H:%M:%S"), style="#008080"
        )  # Teal
        text.append("](", style="white")
        text.append(str(elapsed_time), style="#008080")  # Teal
        text.append(")", style="white")
        if file_name:
            text.append("{", style="white")
            text.append(file_name, style="#008080")  # Teal
            text.append("} ", style="white")
        text.append(message, style="#008080")  # Teal

        self.console.print(text)

        # Clean up the frame reference
        del frame
        del caller_frame


def get_exp_dir(config: dict[str, Any]) -> Path:
    """Get experiment directory based on configuration.

    Args:
        config: Dictionary containing configuration

    Returns:
        Path to experiment directory
    """
    project_dir = Path(config.project_dir or "cat_uncertainty_experiments")
    experiment_name = config.experiment_name or "default_experiment"
    return project_dir / experiment_name


def get_gpu_info_pytorch():
    if torch.cuda.is_available():
        # Get device count
        device_count = torch.cuda.device_count()
        print(f"Found {device_count} GPU(s)")

        for i in range(device_count):
            # Get device properties
            props = torch.cuda.get_device_properties(i)
            # Get memory info
            total_memory = props.total_memory / 1024**2  # Convert to MB
            allocated_memory = torch.cuda.memory_allocated(i) / 1024**2
            cached_memory = torch.cuda.memory_reserved(i) / 1024**2

            print(f"\nGPU {i}: {props.name}")
            print(f"Total memory: {total_memory:.2f} MB")
            print(f"Allocated memory: {allocated_memory:.2f} MB")
            print(f"Cached memory: {cached_memory:.2f} MB")
    else:
        print("No GPU available")
