import numpy as np
from ase.atoms import Atoms
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
    ) -> Data:
        """
        Convert a set of atomic coordinates and cell parameters into a PyG graph.
        """
        if self.pbc and not cell:
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
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_dist,
            three_body_indices=three_body_indices,
            num_bonds_per_atom=num_bonds_per_atom,
            num_angles_per_atom=num_angles_per_atom,
            num_angles_on_central_atom=num_angles_on_central_atom,
            total_num_angles=total_num_angles,
        )

    def convert_ase_atoms(self, atoms: Atoms, **kwargs) -> Data:
        """
        Convert an ASE Atoms object into a PyG graph.
        """
        return self.convert(pos=atoms.positions, cell=atoms.cell, **kwargs)
