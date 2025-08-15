from itertools import permutations

import numpy as np
import torch
from pymatgen.optimization.neighbors import find_points_in_spheres


def compute_fixed_radius_graph(
    pos: np.ndarray,
    cell: np.ndarray | None = None,
    *,
    pbc: bool = True,
    cutoff: float = 5.0,
    numerical_tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a fixed radius graph.

    Args:
        pos (torch.Tensor): The positions of the atoms.
        cell (torch.Tensor | None): The cell parameters.
        pbc (bool): Whether to use periodic boundary conditions.
        cutoff (float): The cutoff radius.
        numerical_tol (float): The numerical tolerance.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            The edge index, distances, and offsets.
    """

    if pbc and not cell:
        raise ValueError("Cell parameters are required when pbc is True.")

    # find_points_in_spheres defined in pymatgen.optimization.neighbors
    # requires continous memory layout, because it calls compiled external C program
    _pos = np.ascontiguousarray(pos)
    _cell = np.ascontiguousarray(cell)
    _pbc = np.array([1] * 3)

    (
        center_indices,
        neighbor_indices,
        offsets,
        distances,
    ) = find_points_in_spheres(
        all_coords=_pos,
        center_coords=_pos,
        r=cutoff,
        pbc=_pbc,
        lattice=_cell,
        tol=numerical_tol,
    )

    center_indices = center_indices.astype(np.int64)
    neighbor_indices = neighbor_indices.astype(np.int64)
    offsets = offsets.astype(np.int64)
    distances = distances.astype(np.float64)

    # exclude self connection
    exclude_self_connection = (center_indices == neighbor_indices) & (
        distances < numerical_tol
    )

    center_indices = center_indices[~exclude_self_connection]
    neighbor_indices = neighbor_indices[~exclude_self_connection]
    offsets = offsets[~exclude_self_connection]
    distances = distances[~exclude_self_connection]

    edge_index = torch.stack(
        [
            torch.from_numpy(center_indices),
            torch.from_numpy(neighbor_indices),
        ],
        dim=0,
    )

    return edge_index, distances, offsets


def compute_threebody_indices(
    pos: np.ndarray,
    edge_index: np.ndarray,
    edge_dist: np.ndarray,
    *,
    three_body_cutoff: float = 3.0,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Given the two-body graph, compute three-body indices within the cutoff.

    Args:
        pos (np.ndarray): Atom positions, shape (num_atoms, 3).
        edge_index (np.ndarray): Two-body graph edges, shape (2, num_edges).
        edge_dist (np.ndarray): Two-body graph distances, shape (num_edges,).
        three_body_cutoff (float): Cutoff radius for bonds forming angles.

    Returns:
        A tuple containing:
        - total_num_angles (np.ndarray): A scalar value.
        - num_bonds_per_atom (np.ndarray): Shape (num_atoms,).
        - num_angles_per_atom (np.ndarray): Shape (num_atoms,).
        - num_angles_on_central_atom (np.ndarray): Shape (num_edges,).
        - three_body_indices (np.ndarray): Shape (num_angles, 2).
    """

    # 0. filter the bonds that are within the two-body cutoff radius
    #    and establish the mapping from the global bond index to the filtered bond index
    #    and vice versa
    mask = edge_dist < three_body_cutoff
    selected_bonds_l2g_map = np.where(mask)[0]
    selected_edge_index = edge_index[:, mask]

    central_indices = selected_edge_index[0]

    # 1. count the number of bonds associated with each central atom
    #   Shape: (num_atoms,)
    num_bonds_per_atom = np.bincount(central_indices, minlength=pos.shape[0])

    # 2. compute the number of ordered angles around each central atom
    #   Shape: (num_atoms,)
    num_angles_per_atom = num_bonds_per_atom * (num_bonds_per_atom - 1)

    # 3. compute the number of angles that a bond is part of
    # Explanation:
    #   If a bond has a central atom i, it will form angles with all of the other atoms
    #   that are connected with the central atom i except itself. Thus, we just need to
    #   subtract 1 from the number of bonds on the central atom.
    #   In addition, for convenience, we make the shape of the
    #   num_angles_on_central_atom the same size as the original edges arrays
    #   (central_indices and neighbor_indices).
    #   Shape: (global_num_bonds,)
    angles_per_selected_bond = num_bonds_per_atom[central_indices] - 1

    num_angles_on_central_atom = np.zeros(len(edge_index[0]), dtype=np.int64)
    num_angles_on_central_atom[selected_bonds_l2g_map] = angles_per_selected_bond

    # 4.1 group the indices of the bonds by the central atom
    #   Shape: (num_atoms, variable num_bonds_per_atom)
    #   The values in the array are the global bond indices.
    bonds_indices_grouped_by_central_atom = [[] for _ in range(pos.shape[0])]
    for bond_index, central_atom_index in enumerate(central_indices):
        bonds_indices_grouped_by_central_atom[central_atom_index].append(
            selected_bonds_l2g_map[bond_index]
        )

    # 4.2 generate the three-body indices of the angles
    # Explanation:
    #   For each central atom, we generate the combinations of the bond indices.
    #   Then, we generate the three-body indices of the angles.
    #   As a result, the three-body indices are indexed by the central atom index.
    #   For example, if the central atom index is 1 and the corresponding values in
    #   the three_body_indices are [3,4], it means that the 3rd and 4th bonds in the
    #   global bond list are the two bonds that form the angle with the central atom 1.
    #   Shape: (num_angles, 2)
    three_body_indices_list = []
    for bond_indices in bonds_indices_grouped_by_central_atom:
        if len(bond_indices) >= 2:
            three_body_indices_list.extend(permutations(bond_indices, 2))

    # FIX: Check if the LIST is empty before converting to a NumPy array.
    if not three_body_indices_list:
        three_body_indices = np.empty((0, 2), dtype=np.int64)
    else:
        three_body_indices = np.array(three_body_indices_list, dtype=np.int64)

    # 5. calculate the total number of angles in this structure
    total_num_angles = np.sum(num_angles_per_atom)

    return (
        total_num_angles,
        num_bonds_per_atom,
        num_angles_per_atom,
        num_angles_on_central_atom,
        three_body_indices,
    )
