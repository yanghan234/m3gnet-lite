import torch
from loguru import logger
from torch import nn
from torch_geometric.data import Data

from .layers.basis import SmoothBesselBasis, SphericalHarmonicAndRadialBasis
from .layers.common import MLP
from .layers.embedding import AtomicEmbedding
from .layers.interaction import MainBlock


class M3GNet(nn.Module):
    def __init__(
        self,
        *,
        num_elements: int = 108,
        num_blocks: int = 4,
        feature_dim: int = 128,
        max_angular_l: int = 4,  # inclusive, i.e. 4 means 0, 1, 2, 3, 4
        max_radial_n: int = 4,  # inclusive, i.e. 4 means 0, 1, 2, 3, 4
        cutoff: float = 5.0,
        enable_three_body: bool = True,
        three_body_cutoff: float = 4.0,
        device: str | torch.device = "cpu",
        **kwargs,
    ):
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
        self.device = device

        # create model components
        self.atomic_embedding = AtomicEmbedding(num_elements, feature_dim)
        self.rbf = SmoothBesselBasis(cutoff, max_radial_n)  # dim = max_radial_n + 1
        self.shrb = SphericalHarmonicAndRadialBasis(max_angular_l, max_radial_n, cutoff)
        self.edge_encoding_mlp = MLP(
            in_dim=max_radial_n + 1,
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

        # move model to device
        self.to(device)

    def forward(self, data: Data, batch: torch.Tensor | None = None) -> torch.Tensor:
        # The edge_indices were computed for each structure, and thus the indices in
        # the three_body_indices all start from 0. Thus, we need to add the cumsum
        # of the number of bonds in the previous structures to the indices in
        # the three_body_indices.
        # Please note, the edge_index attributes of the torch_geometric.data.Data object
        # will be incremented automatically by the DataLoader, so there is no need to
        # add the offset to the edge_index here.
        cumsum_bonds = data.total_num_bonds.cumsum(dim=0).detach()
        offsets = torch.cat(
            [
                torch.zeros(1, device=cumsum_bonds.device, dtype=cumsum_bonds.dtype),
                cumsum_bonds[:-1],
            ]
        )
        # Repeat each offset according to the number of angles in each structure
        offsets = (
            torch.repeat_interleave(offsets, data.total_num_angles)
            .unsqueeze(1)
            .to(data.three_body_indices.device)
        )
        data.three_body_indices_with_offset = data.three_body_indices + offsets

        # Get atomic and edge initial features
        atomic_features = self.atomic_embedding(data.atomic_numbers)  # noqa: F841

        # Get initial edge features and encode them to the required feature dimension
        # so that they can be used in later blocks
        # This encoding is done with a simple MLP
        edge_features = edge_features_0 = self.rbf(data.edge_dist)  # noqa: F841
        edge_features = self.edge_encoding_mlp(edge_features)

        # Get spherical harmonic and radial basis functions
        # These are the representations of the three-body angles,
        # which are used to compute the three-body interactions.
        angle_features = self.shrb(data.norm_ik, data.three_body_cos_angles)  # noqa: F841

        for main_block in self.main_blocks:
            atomic_features, edge_features = main_block(
                data,
                atomic_features,
                edge_features,
                angle_features,
                edge_features_0,
            )

        energy = self.energy_mlp(atomic_features)
        logger.info(f"data.pos.shape: {data.pos.shape}")
        logger.info(f"energy.shape: {energy.shape}")
        logger.info(f"energy: {energy[:10]}")

        return energy
