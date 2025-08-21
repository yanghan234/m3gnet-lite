import torch
from torch import nn
from torch_geometric.data import Data

from .common import MLP, GatedMLP


def envelope_polynomial(x: torch.Tensor, cutoff: float) -> torch.Tensor:
    """
    Envelope polynomial for the cutoff function.
    f(x) = 1 - 6 * (x / cutoff)** 5 + 15 * (x / cutoff)** 4 - 10 * (x / cutoff)** 3

    Args:
        x (torch.Tensor): The input tensor.
        cutoff (float): The cutoff value.

    Returns:
        torch.Tensor: The envelope polynomial.
    """
    return 1 - 6 * (x / cutoff) ** 5 + 15 * (x / cutoff) ** 4 - 10 * (x / cutoff) ** 3


class MainBlock(nn.Module):
    pass


class ThreeBodyInteraction(nn.Module):
    def __init__(
        self,
        *,
        max_angular_l: int,
        max_radial_n: int,
        cutoff: float,
        three_body_cutoff: float,
        feature_dim: int,
    ):
        super().__init__()

        self.max_angular_l = max_angular_l
        self.max_radial_n = max_radial_n
        self.cutoff = cutoff
        self.three_body_cutoff = three_body_cutoff
        self.feature_dim = feature_dim
        self.angle_feature_dim = (max_angular_l + 1) * (max_radial_n + 1)

        self.atom_mlp = MLP(
            in_dim=feature_dim,
            output_dim=self.angle_feature_dim,
            activation="sigmoid",
            bias=True,
        )
        self.edge_gated_mlp = GatedMLP(
            in_dim=self.angle_feature_dim,
            output_dim=[self.feature_dim],
            activation=["swish"],
            bias=True,
        )

    # def forward(self, data: Data, batch: torch.Tensor | None = None) -> torch.Tensor:
    def forward(
        self,
        data: Data,
        atomic_features: torch.Tensor,
        edge_features: torch.Tensor,
        angle_features: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            data (Data): The data object containing the graph structure.
            atomic_features (torch.Tensor): The atomic features.
                Dimension: (num_nodes, feature_dim)
            edge_features (torch.Tensor): The edge features.
                Dimension: (num_edges, feature_dim)
            angle_features (torch.Tensor): The angle features.
                Dimension: (num_angles, feature_dim)
            batch (torch.Tensor, optional): The batch tensor. Defaults to None.

        Returns:
            torch.Tensor: The three-body interaction.
        """

        # apply atomwise MLP
        atomic_filter = self.atom_mlp(atomic_features)

        # Extract edge indices once to avoid repeated indexing
        edge_ij_indices = data.three_body_indices_with_offset[
            :, 0
        ]  # Shape: [num_angles]
        edge_ik_indices = data.three_body_indices_with_offset[
            :, 1
        ]  # Shape: [num_angles]

        # Get atomic filters for the central atoms (more efficient indexing)
        atomic_filter_k = atomic_filter[
            data.edge_index[1, edge_ik_indices]
        ]  # Shape: [num_angles, angle_feature_dim]

        # Compute envelope functions for both edges
        envelope_ij = envelope_polynomial(
            data.edge_dist[edge_ij_indices], self.cutoff
        ).unsqueeze(-1)  # Shape: [num_angles, 1]
        envelope_ik = envelope_polynomial(
            data.edge_dist[edge_ik_indices], self.cutoff
        ).unsqueeze(-1)  # Shape: [num_angles, 1]

        masks = atomic_filter_k * envelope_ij * envelope_ik

        # Vectorized accumulation of masked angle features
        edge_feature_ij_tilde = torch.zeros(
            [data.edge_index.shape[1], self.angle_feature_dim],
            device=data.edge_index.device,
        )

        # Apply masks to angle features (element-wise multiplication)
        masked_angle_features = (
            angle_features * masks
        )  # Shape: [num_angles, angle_feature_dim]

        # Use scatter_add to accumulate masked features for each edge
        edge_feature_ij_tilde.scatter_add_(
            dim=0,
            index=edge_ij_indices.unsqueeze(-1).expand(-1, self.angle_feature_dim),
            src=masked_angle_features,
        )

        return edge_features + self.edge_gated_mlp(edge_feature_ij_tilde)
