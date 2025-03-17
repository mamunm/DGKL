"""This module defines the SOAP transform for graph data."""

import pickle
from pathlib import Path

import lmdb
import numpy as np
import torch
from ase import Atoms
from ase.data import chemical_symbols
from dscribe.descriptors import SOAP
from torch_geometric.data import Data

from ..dataclass.data_config import SOAPTransformParams


def convert_to_atoms(data) -> Atoms:
    """
    Convert PyG Data object to ASE Atoms object.

    Args:
        data: PyTorch Geometric Data object

    Returns:
        Atoms: ASE Atoms object
    """
    pos = data.pos.numpy()
    numbers = data.atomic_numbers.numpy()
    cell = data.cell.numpy()[0] if hasattr(data, "cell") else None
    pbc = data.pbc if hasattr(data, "pbc") else True
    energy = data.energy if hasattr(data, "energy") else None

    atoms = Atoms(positions=pos, numbers=numbers, cell=cell, pbc=pbc)
    atoms.info["energy"] = energy

    return atoms


def get_species_from_lmdb(db_paths: list[str | Path]) -> set[str]:
    """
    Utility function to get the unique species from an LMDB file.

    Args:
        db_path (list[str | Path]): Path to the LMDB file.

    Returns:
        Set[str]: A set of unique species found in the LMDB file.
    """
    atomic_numbers_set = set()

    for db_path in db_paths:
        env = lmdb.open(str(db_path), subdir=False, readonly=True, lock=False)
        with env.begin() as txn:
            cursor = txn.cursor()
            for _, value in cursor:
                data = pickle.loads(value)
                structure = convert_to_atoms(data)
                atomic_numbers_set.update([str(z) for z in structure.numbers])
    atomic_numbers_list = list(atomic_numbers_set)
    species_list = [chemical_symbols[int(z)] for z in atomic_numbers_list]
    return species_list


class SOAPFeatureExtractor:
    """
    Extracts SOAP features from structures stored in an LMDB database.

    Args:
        soap_params (SOAPParams): Parameters for the SOAP descriptor.
    """

    def __init__(
        self,
        soap_params: SOAPTransformParams,
        db_paths: list[str | Path] | None = None,
    ):
        self.soap_params = soap_params
        self.db_paths = db_paths
        species = list(get_species_from_lmdb(db_paths))
        self.soap = SOAP(
            species=species,
            r_cut=self.soap_params.rcut,
            n_max=self.soap_params.nmax,
            l_max=self.soap_params.lmax,
            sigma=self.soap_params.sigma,
            periodic=self.soap_params.periodic,
            sparse=self.soap_params.sparse,
            average=self.soap_params.average,
        )

    def transform(self, data: "Data") -> np.ndarray:
        """
        Transform PyG Data object to SOAP features.

        Args:
            data (Data): PyTorch Geometric Data object representing a
                molecular structure.

        Returns:
            np.ndarray: SOAP features for the given structure.
        """
        structure = convert_to_atoms(data)
        features = torch.from_numpy(self.soap.create(structure))
        if self.soap_params.average != "off":
            features = features.reshape(1, -1)

        data = Data(
            x=features.to(data.pos.dtype),
            edge_index=data.edge_index.to(data.atomic_numbers.dtype),
            energy=data.energy,
        )
        return data
