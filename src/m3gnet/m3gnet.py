"""The core M3GNet model architecture."""

import torch
from torch import nn
from torch_geometric.data import Data

from .layers.basis import SmoothBesselBasis, SphericalHarmonicAndRadialBasis
from .layers.common import MLP
from .layers.embedding import AtomicEmbedding
from .layers.interaction import MainBlock


class M3GNet(nn.Module):
    """Core model architecture of M3GNet.

    Args:
        num_elements (int): Number of unique elements in the dataset. Default is 108.
        num_blocks (int): Number of interaction blocks. Default is 4.
        feature_dim (int): Dimension of the atomic features. Default is 128.
        max_angular_l (int): Maximum angular momentum. Default is 4.
        max_radial_n (int): Maximum number of radial basis functions. Default is 4.
        cutoff (float): Cutoff distance for two-body interactions. Default is 5.0.
        enable_three_body (bool): Whether to enable three-body interactions.
            Default is True.
        three_body_cutoff (float): Cutoff distance for three-body interactions.
            Default is 4.0.
    """

    def __init__(
        self,
        *,
        num_elements: int = 108,
        num_blocks: int = 4,
        feature_dim: int = 128,
        max_angular_l: int = 4,  # exclusive, i.e. 4 means 0, 1, 2, 3
        max_radial_n: int = 4,  # exclusive, i.e. 4 means 0, 1, 2, 3
        cutoff: float = 5.0,
        enable_three_body: bool = True,
        three_body_cutoff: float = 4.0,
        **kwargs,
    ):
        """Initialize the M3GNet model."""
        super().__init__()

        # save model parameters
        self.num_elements = num_elements
        self.num_blocks = num_blocks
        self.feature_dim = feature_dim
        self.max_angular_l = max_angular_l
        self.max_radial_n = max_radial_n
        self.cutoff = cutoff
        self.enable_three_body = enable_three_body
        self.three_body_cutoff = three_body_cutoff

        # create model components
        self.atomic_embedding = AtomicEmbedding(num_elements, feature_dim)
        self.rbf = SmoothBesselBasis(cutoff, max_radial_n)  # dim = max_radial_n
        self.shrb = SphericalHarmonicAndRadialBasis(max_angular_l, max_radial_n, cutoff)
        self.edge_encoding_mlp = MLP(
            in_dim=max_radial_n,
            output_dim=[feature_dim],
            activation=["swish"],
            bias=True,
        )
        self.main_blocks = nn.ModuleList(
            [
                MainBlock(
                    max_angular_l=max_angular_l,
                    max_radial_n=max_radial_n,
                    cutoff=cutoff,
                    three_body_cutoff=three_body_cutoff,
                    feature_dim=feature_dim,
                )
                for _ in range(num_blocks)
            ]
        )
        self.energy_mlp = MLP(
            in_dim=feature_dim,
            output_dim=[feature_dim, feature_dim, 1],
            activation=["swish", "swish", None],
        )

    def forward(self, data: Data, batch: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass of the M3GNet model."""
        # 1. Add offset to the three_body_indices to make them global indices
        # The edge_indices were computed for each structure, and thus the indices in
        # the three_body_indices all start from 0. Thus, we need to add the cumsum
        # of the number of bonds in the previous structures to the indices in
        # the three_body_indices.
        # Please note, the edge_index attributes of the torch_geometric.data.Data object
        # will be incremented automatically by the DataLoader, so there is no need to
        # add the offset to the edge_index here.
        cumsum_edges = data.total_num_edges.cumsum(dim=0).detach()
        offsets = torch.cat(
            [
                torch.zeros(1, device=cumsum_edges.device, dtype=cumsum_edges.dtype),
                cumsum_edges[:-1],
            ]
        )
        # Repeat each offset according to the number of angles in each structure
        offsets = torch.repeat_interleave(offsets, data.total_num_angles).unsqueeze(1)
        data.three_body_indices_with_offset = data.three_body_indices + offsets

        # 2. Compute the three-body angles, edge_distances, etc.
        batch = torch.repeat_interleave(
            torch.arange(
                data.total_num_edges.shape[0], device=data.total_num_edges.device
            ),
            data.total_num_edges,
        )
        edge_offsets = torch.einsum(
            "ei, eij->ej", data.edge_offsets, data.cell.reshape(-1, 3, 3)[batch]
        )
        edge_vec = (
            data.pos[data.edge_index[1]] - data.pos[data.edge_index[0]] + edge_offsets
        )
        edge_dist = torch.norm(edge_vec, dim=1)
        data.edge_dist = edge_dist

        edge_ij_indices = data.three_body_indices_with_offset[:, 0]
        edge_ik_indices = data.three_body_indices_with_offset[:, 1]

        batch = torch.repeat_interleave(
            torch.arange(
                data.total_num_atoms.shape[0], device=data.total_num_edges.device
            ),
            data.total_num_edges,
        )
        edge_offsets = torch.einsum(
            "ei, eij->ej", data.edge_offsets, data.cell.reshape(-1, 3, 3)[batch]
        )

        vec_ij = edge_vec[edge_ij_indices]
        vec_ik = edge_vec[edge_ik_indices]
        norm_ik = edge_dist[edge_ik_indices]
        data.norm_ik = norm_ik

        cos_angle = torch.sum(vec_ij * vec_ik, dim=1) / (
            torch.norm(vec_ij, dim=1) * torch.norm(vec_ik, dim=1)
        )
        data.three_body_cos_angles = cos_angle

        # 3. Get initial embeddings of atoms and edges.
        atomic_features = self.atomic_embedding(data.atomic_numbers)
        edge_features = edge_features_0 = self.rbf(data.edge_dist)
        edge_features = self.edge_encoding_mlp(edge_features)

        # 4. Get spherical harmonic and radial basis representations
        # of three-body angles, which are used to compute the three-body interactions.
        angle_features = self.shrb(data.norm_ik, data.three_body_cos_angles)

        # 5. Message passing blocks with three-body interactions inside each block.
        for main_block in self.main_blocks:
            atomic_features, edge_features = main_block(
                data,
                atomic_features,
                edge_features,
                angle_features,
                edge_features_0,
            )

        # 6. Read out the energy per atom from final atomic features.
        energy_per_atom = self.energy_mlp(atomic_features).squeeze(-1)

        batch = torch.repeat_interleave(
            torch.arange(
                data.total_num_atoms.shape[0], device=data.total_num_atoms.device
            ),
            data.total_num_atoms,
        )
        return torch.scatter_add(
            torch.zeros(
                data.total_num_atoms.shape[0], device=data.total_num_atoms.device
            ),
            dim=0,
            index=batch,
            src=energy_per_atom,
        )
