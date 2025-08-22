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
        three_body_cutoff: float | None = 4.0,
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
    ) -> Data:
        """
        Convert a set of atomic coordinates and cell parameters into a PyG graph.
        """
        if self.pbc and cell is None:
            raise ValueError("Cell parameters are required when pbc is True.")

        edge_index, edge_dist, edge_offsets = compute_fixed_radius_graph(
            pos=pos, cell=cell, pbc=self.pbc, cutoff=self.cutoff
        )

        data = Data(
            total_num_atoms=torch.tensor(pos.shape[0], dtype=torch.long),
            total_num_edges=torch.tensor(edge_index.shape[1], dtype=torch.long),
            atomic_numbers=torch.tensor(atomic_numbers, dtype=torch.long),
            pos=torch.tensor(pos, dtype=torch.float32, requires_grad=True),
            cell=torch.tensor(cell, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_dist=torch.tensor(edge_dist, dtype=torch.float32),
            edge_offsets=torch.tensor(edge_offsets, dtype=torch.float32),
        )

        # check if all edge_dist are non-zero
        if sum(np.where(edge_dist < 1e-6, 1, 0)) > 0:
            raise ValueError("At least one edge distance is zero")

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
            data.three_body_indices = torch.tensor(three_body_indices, dtype=torch.long)
            data.num_bonds_per_atom = torch.tensor(num_bonds_per_atom, dtype=torch.long)
            data.num_angles_per_atom = torch.tensor(
                num_angles_per_atom, dtype=torch.long
            )
            data.num_angles_on_central_atom = torch.tensor(
                num_angles_on_central_atom, dtype=torch.long
            )
            data.total_num_angles = torch.tensor(total_num_angles, dtype=torch.long)

        return data

    def convert_ase_atoms(self, atoms: Atoms, **kwargs) -> Data:
        """
        Convert an ASE Atoms object into a PyG graph.
        """
        return self.convert(
            pos=atoms.positions,
            cell=atoms.cell,
            atomic_numbers=atoms.get_atomic_numbers(),
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
