import torch
from loguru import logger
from torch import nn
from torch_geometric.data import Data

from .layers.common import MLP
from .layers.embedding import AtomicEmbedding, SmoothBesselBasis


class M3GNet(nn.Module):
    def __init__(
        self,
        *,
        num_elements: int = 108,
        num_blocks: int = 4,
        feature_dim: int = 128,
        max_angular_l: int = 4,
        max_radial_n: int = 4,
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
        self.rbf = SmoothBesselBasis(cutoff, max_radial_n)
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

    def forward(self, data: Data) -> torch.Tensor:
        # get atomic and edge initial features
        atomic_features = self.atomic_embedding(data.atomic_numbers)
        edge_features_0 = self.rbf(data.edge_dist)
        logger.info(atomic_features.shape, edge_features_0.shape)
        # print(data.edge_dist[:5])
        # print(edge_features_0[:5])

        # energy = self.energy_mlp(atomic_features)
        # return energy
