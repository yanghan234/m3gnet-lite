import torch
from torch import nn
from torch_geometric.data import Data

from .layers.basis import SmoothBesselBasis, SphericalHarmonicAndRadialBasis
from .layers.common import MLP
from .layers.embedding import AtomicEmbedding


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
        three_body_cutoff: float = 3.0,
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
        # self.main_blocks = nn.ModuleList(
        #     [MainBlock(feature_dim) for _ in range(num_blocks)]
        # )
        # self.graph_convolutions = nn.ModuleList(
        #     [GraphConvolution(feature_dim) for _ in range(num_blocks)]
        # )
        # self.mlp = MLP(in_dim=feature_dim, output_dim=feature_dim, activation=")
        self.energy_mlp = MLP(
            in_dim=feature_dim,
            output_dim=[feature_dim, feature_dim, 1],
            activation=["swish", "swish", None],
        )

        # move model to device
        self.to(device)

    def forward(self, data: Data, batch: torch.Tensor | None = None) -> torch.Tensor:
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
        shrb = self.shrb(data.norm_ik, data.three_body_cos_angles)  # noqa: F841

        # energy = self.energy_mlp(atomic_features)
        # return energy
