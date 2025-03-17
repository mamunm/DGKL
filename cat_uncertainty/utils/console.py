"""Simple console that writes to a log file with timestamps."""

from datetime import datetime
from pathlib import Path


class Console:
    """Simple console that writes to a log file with timestamps."""

    def __init__(self, log_file: str | Path):
        log_path = Path(log_file)
        if log_path.exists():
            log_path.unlink()
        self.log_file = open(log_path, "a", encoding="utf-8")
        self.start_time = datetime.now()

    def _get_elapsed_time(self) -> str:
        """Get elapsed time since console creation."""
        elapsed = datetime.now() - self.start_time
        total_seconds = int(elapsed.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _write_with_timestamp(self, text: str) -> None:
        """Write text with timestamp to log file."""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed = self._get_elapsed_time()
        self.log_file.write(f"[{current_time}]({elapsed}) {text}\n")
        self.log_file.flush()

    def print(self, *args, **kwargs) -> None:
        """Write message to file with timestamp."""
        message = " ".join(str(arg) for arg in args)
        self._write_with_timestamp(message)

    def __del__(self) -> None:
        if hasattr(self, "log_file"):
            self.log_file.close()
