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

        edge_index, edge_dist, edge_offsets_coeff = compute_fixed_radius_graph(
            pos=pos, cell=cell, pbc=self.pbc, cutoff=self.cutoff
        )
        edge_offsets = np.einsum("ij, jk->ik", edge_offsets_coeff, cell)

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

            # compute the three-body angles (vectorized)
            if len(three_body_indices) > 0:
                # Extract edge indices for all three-body interactions at once
                edge_ij_indices = three_body_indices[:, 0]  # Shape: [num_angles]
                edge_ik_indices = three_body_indices[:, 1]  # Shape: [num_angles]

                # Get atom indices for all interactions
                atom_i_indices = edge_index[0, edge_ij_indices]  # Shape: [num_angles]
                atom_j_indices = edge_index[1, edge_ij_indices]  # Shape: [num_angles]
                atom_k_indices = edge_index[1, edge_ik_indices]  # Shape: [num_angles]

                # Verify that atom_i is the same for both edges (central atom)
                diff = np.linalg.norm(
                    abs(atom_i_indices - edge_index[0, edge_ik_indices])
                )
                if diff > 1e-8:
                    raise ValueError("Central atom mismatch")

                # Get positions for all atoms at once
                pos_i = pos[atom_i_indices]  # Shape: [num_angles, 3]
                pos_j = (
                    pos[atom_j_indices] + edge_offsets[edge_ij_indices]
                )  # Shape: [num_angles, 3]
                pos_k = (
                    pos[atom_k_indices] + edge_offsets[edge_ik_indices]
                )  # Shape: [num_angles, 3]

                # Compute vectors from central atom to outer atoms
                vec_ij = pos_j - pos_i  # Shape: [num_angles, 3]
                vec_ik = pos_k - pos_i  # Shape: [num_angles, 3]

                # Compute norms of vectors
                norm_ij = np.linalg.norm(
                    vec_ij, axis=1, keepdims=True
                )  # Shape: [num_angles, 1]
                norm_ik = np.linalg.norm(
                    vec_ik, axis=1, keepdims=True
                )  # Shape: [num_angles, 1]

                # check if all norms are non-zero
                if (
                    sum(np.where(norm_ij < 1e-6, 1, 0)) > 0
                    or sum(np.where(norm_ik < 1e-6, 1, 0)) > 0
                ):
                    raise ValueError("At least one norm is zero")

                # Avoid division by zero
                norm_ij = np.where(norm_ij < 1e-8, 1.0, norm_ij)
                norm_ik = np.where(norm_ik < 1e-8, 1.0, norm_ik)

                # Normalize vectors
                vec_ij_norm = vec_ij / norm_ij  # Shape: [num_angles, 3]
                vec_ik_norm = vec_ik / norm_ik  # Shape: [num_angles, 3]

                # Compute dot products for all angles at once
                cos_ijk = np.sum(
                    vec_ij_norm * vec_ik_norm, axis=1
                )  # Shape: [num_angles]

                # Clamp to avoid numerical issues with arccos
                cos_ijk = np.clip(cos_ijk, -1.0 + 1e-8, 1.0 - 1e-8)

                # Compute angles for all interactions at once
                angle_ijk = np.arccos(cos_ijk)  # Shape: [num_angles]

                three_body_cos_angles = cos_ijk
                three_body_angles = angle_ijk
            else:
                three_body_cos_angles = np.array([], dtype=np.float64)
                three_body_angles = np.array([], dtype=np.float64)
        else:
            total_num_angles = 0
            num_bonds_per_atom = np.zeros(pos.shape[0], dtype=np.int64)
            num_angles_per_atom = np.zeros(pos.shape[0], dtype=np.int64)
            num_angles_on_central_atom = np.zeros(edge_index.shape[1], dtype=np.int64)
            three_body_indices = np.empty((0, 2), dtype=np.int64)
            three_body_cos_angles = np.array([], dtype=np.float64)

        return Data(
            atomic_numbers=torch.tensor(atomic_numbers, dtype=torch.long)
            if atomic_numbers is not None
            else torch.zeros(pos.shape[0], dtype=torch.long),
            pos=torch.tensor(pos, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_dist=torch.tensor(edge_dist, dtype=torch.float32),
            edge_offsets=torch.tensor(edge_offsets, dtype=torch.float32),
            edge_offsets_coeff=torch.tensor(edge_offsets_coeff, dtype=torch.float32),
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
            three_body_cos_angles=torch.tensor(
                three_body_cos_angles, dtype=torch.float32
            ),
            three_body_angles=torch.tensor(three_body_angles, dtype=torch.float32),
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
