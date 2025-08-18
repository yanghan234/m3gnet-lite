import numpy as np
import torch
from ase.atoms import Atoms
from pymatgen.core.structure import Structure
from torch_geometric.data import Data

from .utils import compute_fixed_radius_graph, compute_threebody_indices


class GraphConverter:
    """
    Convert a set of atomic coordinates and cell parameters into a PyG graph.

    Args:
        cutoff: Cutoff radius for the graph construction. Default is 5.0.
        pbc: Whether to use periodic boundary conditions. Default is True.
        three_body_cutoff: Cutoff radius for three-body interactions. Default is None.
    """

    def __init__(
        self,
        *,
        cutoff: float = 5.0,
        pbc: bool = True,
        three_body_cutoff: float | None = 3.0,
    ):
        self.cutoff = cutoff
        self.pbc = pbc
        self.three_body_cutoff = three_body_cutoff

    def convert(
        self,
        *,
        pos: np.ndarray,
        cell: np.ndarray | None = None,
        atomic_numbers: np.ndarray | None = None,
        potential_energy: float | None = None,
        forces: np.ndarray | None = None,
        stress: np.ndarray | None = None,
    ) -> Data:
        """
        Convert a set of atomic coordinates and cell parameters into a PyG graph.
        """
        if self.pbc and cell is None:
            raise ValueError("Cell parameters are required when pbc is True.")

        edge_index, edge_dist, _ = compute_fixed_radius_graph(
            pos=pos, cell=cell, pbc=self.pbc, cutoff=self.cutoff
        )

        if self.three_body_cutoff:
            (
                total_num_angles,
                num_bonds_per_atom,
                num_angles_per_atom,
                num_angles_on_central_atom,
                three_body_indices,
            ) = compute_threebody_indices(
                pos=pos,
                edge_index=edge_index,
                edge_dist=edge_dist,
                three_body_cutoff=self.three_body_cutoff,
            )
        else:
            total_num_angles = 0
            num_bonds_per_atom = np.zeros(pos.shape[0], dtype=np.int64)
            num_angles_per_atom = np.zeros(pos.shape[0], dtype=np.int64)
            num_angles_on_central_atom = np.zeros(edge_index.shape[1], dtype=np.int64)
            three_body_indices = np.empty((0, 2), dtype=np.int64)

        return Data(
            atomic_numbers=torch.tensor(atomic_numbers, dtype=torch.long),
            pos=torch.tensor(pos, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_dist=torch.tensor(edge_dist, dtype=torch.float32),
            three_body_indices=torch.tensor(three_body_indices, dtype=torch.long),
            num_bonds_per_atom=torch.tensor(num_bonds_per_atom, dtype=torch.long),
            num_angles_per_atom=torch.tensor(num_angles_per_atom, dtype=torch.long),
            num_angles_on_central_atom=torch.tensor(
                num_angles_on_central_atom, dtype=torch.long
            ),
            total_num_angles=torch.tensor(total_num_angles, dtype=torch.long),
            potential_energy=torch.tensor(potential_energy, dtype=torch.float32)
            if potential_energy is not None
            else None,
            forces=torch.tensor(forces, dtype=torch.float32)
            if forces is not None
            else None,
            stress=torch.tensor(stress, dtype=torch.float32)
            if stress is not None
            else None,
        )

    def convert_ase_atoms(self, atoms: Atoms, **kwargs) -> Data:
        """
        Convert an ASE Atoms object into a PyG graph.
        """
        return self.convert(
            pos=atoms.positions,
            cell=atoms.cell,
            atomic_numbers=atoms.get_atomic_numbers(),
            potential_energy=atoms.get_potential_energy(),
            forces=atoms.get_forces(),
            stress=atoms.get_stress(),
            **kwargs,
        )

    def convert_pymatgen_structure(self, structure: Structure, **kwargs) -> Data:
        """
        Convert a PyMatGen Structure object into a PyG graph.
        """
        return self.convert(
            pos=structure.cart_coords,
            cell=structure.lattice.matrix,
            atomic_numbers=structure.atomic_numbers,
            **kwargs,
        )
