"""Dataset for loading graph data from LMDB files."""

import pickle
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path

import lmdb
from torch.utils.data import Dataset
from torch_geometric.data import Data

# Suppress the specific FutureWarning from torch.load
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="torch.storage"
)


class LMDBDataset(Dataset):
    """
    A dataset that loads data from LMDB files containing serialized
    PyTorch Geometric Data objects.

    Args:
        db_path (str | Path): Directory containing LMDB files
        lmdb_files (str | Sequence[str] | None): LMDB files to load. Can be:
            - None: Load all .lmdb files in db_path
            - str: Load single file from db_path/lmdb_files
            - Sequence[str]: Load multiple files from db_path/lmdb_files[i]
        transform (callable, optional): A function/transform that takes in a
            torch_geometric.data.Data object and returns a transformed version.
    """

    def __init__(
        self,
        db_path: str | Path,
        lmdb_files: str | Sequence[str] | None = None,
        transform: Callable = None,
    ):
        super().__init__()
        self.db_path = Path(db_path)
        if not self.db_path.is_dir():
            raise ValueError(f"db_path {db_path} must be a directory")

        self.lmdb_files = lmdb_files
        self.transform = transform
        self.db_paths: list[Path] = []
        self.envs = []
        self._keys_per_db = []
        self._cumulative_sizes = []
        self._len = 0

        # Find all LMDB databases and initialize environments
        self._find_databases()
        self._initialize_environments()

    def _find_databases(self):
        """
        Find all LMDB databases based on input configuration.
        Handles three cases:
        1. lmdb_files is None: Find all .lmdb files in db_path
        2. lmdb_files is str: Use single file from db_path/lmdb_files
        3. lmdb_files is sequence: Use multiple files from db_path/lmdb_files[i]
        """
        if self.lmdb_files is None:
            self.db_paths = list(self.db_path.glob("*.lmdb"))
        elif isinstance(self.lmdb_files, str):
            path = self.db_path / self.lmdb_files
            if not path.exists():
                raise ValueError(f"LMDB file not found: {path}")
            self.db_paths = [path]
        elif isinstance(self.lmdb_files, list | tuple):
            self.db_paths = []
            for fname in self.lmdb_files:
                path = self.db_path / fname
                if not path.exists():
                    raise ValueError(f"LMDB file not found: {path}")
                self.db_paths.append(path)
        else:
            raise ValueError(
                f"Expected None, str, or Sequence[str], "
                f"got {type(self.lmdb_files)}"
            )

        if not self.db_paths:
            raise RuntimeError(f"No LMDB databases found in {self.db_path}")

        self.db_paths.sort()

    def _initialize_environments(self):
        """Initialize all LMDB environments and collect keys."""
        try:
            for db_path in self.db_paths:
                env = lmdb.open(
                    str(db_path),
                    subdir=False,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False,
                    max_readers=1,
                )
                self.envs.append(env)

                with env.begin() as txn:
                    keys = list(txn.cursor().iternext(values=False))
                    self._keys_per_db.append(keys)

                    current_size = len(keys)
                    if not self._cumulative_sizes:
                        self._cumulative_sizes.append(current_size)
                    else:
                        self._cumulative_sizes.append(
                            self._cumulative_sizes[-1] + current_size
                        )

            self._len = (
                self._cumulative_sizes[-1] if self._cumulative_sizes else 0
            )

        except Exception as e:
            self.close_environments()
            raise e

    def _get_db_index(self, idx: int) -> tuple[int, int]:
        """
        Convert global index to (db_index, local_index).

        Args:
            idx (int): Global index

        Returns:
            tuple[int, int]: (db_index, local_index)
        """
        if idx < 0 or idx >= self._len:
            raise IndexError(f"Index {idx} out of range")

        db_idx = 0
        while (
            db_idx < len(self._cumulative_sizes)
            and idx >= self._cumulative_sizes[db_idx]
        ):
            db_idx += 1

        local_idx = idx
        if db_idx > 0:
            local_idx = idx - self._cumulative_sizes[db_idx - 1]

        return db_idx, local_idx

    def __len__(self) -> int:
        """Returns the total number of examples across all databases."""
        return self._len

    def __getitem__(self, idx: int) -> Data:
        """Gets the data object at index idx."""
        db_idx, local_idx = self._get_db_index(idx)
        key = self._keys_per_db[db_idx][local_idx]

        with self.envs[db_idx].begin() as txn:
            raw_data = txn.get(key)
            data = pickle.loads(raw_data)

            if self.transform is not None:
                data = self.transform(data)

        return data

    def close_environments(self):
        """Safely close all LMDB environments."""
        for env in self.envs:
            env.close()
        self.envs.clear()
        self._keys_per_db.clear()
        self._cumulative_sizes.clear()
        self._len = 0

    def __del__(self):
        """Cleanup LMDB environments on deletion."""
        self.close_environments()
