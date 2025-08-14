import numpy as np
import torch
from pymatgen.optimization.neighbors import find_points_in_spheres


def compute_fixed_radius_graph(
    pos: torch.Tensor,
    cell: torch.Tensor | None = None,
    pbc: bool = True,
    cutoff: float = 5.0,
) -> torch.Tensor:
    """
    Compute a fixed radius graph.

    Args:
        pos (torch.Tensor): The positions of the atoms.
        cell (torch.Tensor | None): The cell parameters.
        pbc (bool): Whether to use periodic boundary conditions.
        cutoff (float): The cutoff radius.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            The edge index, distances, and offsets.
    """

    if pbc and not cell:
        raise ValueError("Cell parameters are required when pbc is True.")

    # find_points_in_spheres defined in pymatgen.optimization.neighbors
    # requires continous memory layout, because it calls compiled external C program
    _pos = np.ascontiguousarray(pos.cpu().numpy())
    _cell = np.ascontiguousarray(cell.cpu().numpy())
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
    )

    center_indices = center_indices.astype(np.int64)
    neighbor_indices = neighbor_indices.astype(np.int64)
    offsets = offsets.astype(np.int64)
    distances = distances.astype(np.float64)

    edge_index = torch.stack(
        [
            torch.from_numpy(center_indices),
            torch.from_numpy(neighbor_indices),
        ],
        dim=0,
    )

    return edge_index, distances, offsets
