"""The core M3GNet model architecture."""

import torch
from torch import nn
from torch_geometric.data import Data

from .layers.basis import SmoothBesselBasis, SphericalHarmonicAndRadialBasis
from .layers.common import MLP
from .layers.embedding import AtomicEmbedding
from .layers.interaction import MainBlock
from .layers.scaling import AtomicScaling


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
        self.normalizer = AtomicScaling(num_elements, trainable=True)

    def forward(self, data: Data, batch: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass of the M3GNet model."""
        # 1. Calculate offsets for three-body indices to make them global
        cumsum_edges = data.total_num_edges.cumsum(dim=0).detach()
        offsets = torch.cat(
            [
                torch.zeros(1, device=cumsum_edges.device, dtype=cumsum_edges.dtype),
                cumsum_edges[:-1],
            ]
        )
        offsets = torch.repeat_interleave(offsets, data.total_num_angles).unsqueeze(1)
        three_body_indices_with_offset = data.three_body_indices + offsets

        # 2. Compute edge and angle properties from graph structure
        batch_map = torch.repeat_interleave(
            torch.arange(
                data.total_num_edges.shape[0], device=data.total_num_edges.device
            ),
            data.total_num_edges,
        )
        edge_offsets = torch.einsum(
            "ei, eij->ej", data.edge_offsets, data.cell.reshape(-1, 3, 3)[batch_map]
        )
        edge_vec = (
            data.pos[data.edge_index[1]] - data.pos[data.edge_index[0]] + edge_offsets
        )
        edge_dist = torch.norm(edge_vec, dim=1)

        edge_ij_indices = three_body_indices_with_offset[:, 0]
        edge_ik_indices = three_body_indices_with_offset[:, 1]

        vec_ij = edge_vec[edge_ij_indices]
        vec_ik = edge_vec[edge_ik_indices]
        norm_ik = edge_dist[edge_ik_indices]

        cos_angle = torch.sum(vec_ij * vec_ik, dim=1) / (
            torch.norm(vec_ij, dim=1) * torch.norm(vec_ik, dim=1)
        )
        # Numerical noise near ±1 would otherwise make acos unstable.
        cos_angle = cos_angle.clamp(
            -1.0 + torch.finfo(cos_angle.dtype).eps,
            1.0 - torch.finfo(cos_angle.dtype).eps,
        )
        theta = torch.acos(cos_angle)

        # 3. Get initial embeddings of atoms and edges
        atomic_features = self.atomic_embedding(data.atomic_numbers)
        edge_features_0 = self.rbf(edge_dist)
        edge_features = self.edge_encoding_mlp(edge_features_0)

        # 4. Get spherical harmonic and radial basis representations of angles
        angle_features = self.shrb(norm_ik, theta)

        # 5. Message passing blocks
        for main_block in self.main_blocks:
            atomic_features, edge_features = main_block(
                atomic_features,
                edge_features,
                angle_features,
                edge_features_0,
                three_body_indices_with_offset,
                data.edge_index,
                edge_dist,
            )

        # 6. Read out the energy per atom from final atomic features
        energy_per_atom = self.energy_mlp(atomic_features).squeeze(-1)
        energy_per_atom_denormalized = self.normalizer(
            atomic_numbers=data.atomic_numbers,
            atomic_energies=energy_per_atom,
        )

        batch_map = torch.repeat_interleave(
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
            index=batch_map,
            src=energy_per_atom_denormalized,
        )
