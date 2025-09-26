"""This module implements the interaction layers for M3GNet."""

import torch
from torch import nn

from .common import MLP, GatedMLP, LinearLayer


def envelope_polynomial(x: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Envelope polynomial for the cutoff function.

    f(x) = 1 - 6 * (x / cutoff)** 5 + 15 * (x / cutoff)** 4 - 10 * (x / cutoff)** 3

    Args:
        x (torch.Tensor): The input tensor.
        cutoff (float): The cutoff value.

    Returns:
        torch.Tensor: The envelope polynomial.
    """
    results = 1 - 6 * (x / cutoff) ** 5 + 15 * (x / cutoff) ** 4 - 10 * (x / cutoff) ** 3
    return torch.clamp(results, min=0.0)

class MainBlock(nn.Module):
    """The main interaction block for M3GNet."""

    def __init__(
        self,
        *,
        max_angular_l: int,
        max_radial_n: int,
        cutoff: float,
        three_body_cutoff: float,
        feature_dim: int,
    ):
        """Initialize the MainBlock class.

        Args:
            max_angular_l (int): The maximum angular momentum.
            max_radial_n (int): The maximum number of radial basis functions.
            cutoff (float): The cutoff radius.
            three_body_cutoff (float): The three-body cutoff radius.
            feature_dim (int): The feature dimension.
        """
        super().__init__()

        self.max_angular_l = max_angular_l
        self.max_radial_n = max_radial_n
        self.cutoff = cutoff
        self.three_body_cutoff = three_body_cutoff
        self.feature_dim = feature_dim
        self.angle_feature_dim = max_angular_l * max_radial_n

        self.edge_update_gated_mlp = GatedMLP(
            in_dim=3 * self.feature_dim,
            output_dim=[self.feature_dim, self.feature_dim],
            activation=["swish", "swish"],
            bias=True,
        )
        self.initial_edge_linear_1 = LinearLayer(
            in_dim=self.max_radial_n,
            out_dim=self.feature_dim,
            bias=True,
        )

        self.atom_update_gated_mlp = GatedMLP(
            in_dim=3 * self.feature_dim,
            output_dim=[self.feature_dim, self.feature_dim],
            activation=["swish", "swish"],
            bias=True,
        )
        self.initial_edge_linear_2 = LinearLayer(
            in_dim=self.max_radial_n,
            out_dim=self.feature_dim,
            bias=True,
        )
        self.three_body_interaction = ThreeBodyInteraction(
            max_angular_l=max_angular_l,
            max_radial_n=max_radial_n,
            cutoff=cutoff,
            three_body_cutoff=three_body_cutoff,
            feature_dim=feature_dim,
        )

    def forward(
        self,
        atomic_features: torch.Tensor,
        edge_features: torch.Tensor,
        angle_features: torch.Tensor,
        initial_edge_features: torch.Tensor,
        three_body_indices_with_offset: torch.Tensor,
        edge_index: torch.Tensor,
        edge_dist: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for the main interaction block.

        Args:
            atomic_features (torch.Tensor): The atomic features.
            edge_features (torch.Tensor): The edge features.
            angle_features (torch.Tensor): The angle features.
            initial_edge_features (torch.Tensor): The initial edge features.
            three_body_indices_with_offset (torch.Tensor): Three-body indices.
            edge_index (torch.Tensor): Edge indices.
            edge_dist (torch.Tensor): Edge distances.

        Returns:
            torch.Tensor: The updated atomic and edge features.
        """
        edge_features = self.three_body_interaction(
            atomic_features,
            edge_features,
            angle_features,
            three_body_indices_with_offset,
            edge_index,
            edge_dist,
        )

        # vectorize the edge update
        concat_features = torch.cat(
            [
                atomic_features[edge_index[0]],  # (num_edges, feature_dim)
                atomic_features[edge_index[1]],  # (num_edges, feature_dim)
                edge_features,  # (num_edges, feature_dim)
            ],
            dim=-1,
        )  # (num_edges, 3*feature_dim)

        edge_updates = self.edge_update_gated_mlp(
            concat_features
        ) * self.initial_edge_linear_1(
            initial_edge_features
        )  # (num_edges, feature_dim)

        edge_features = edge_features + edge_updates

        concat_features = torch.cat(
            [
                atomic_features[edge_index[0]],  # (num_edges, feature_dim)
                atomic_features[edge_index[1]],  # (num_edges, feature_dim)
                edge_features,  # (num_edges, feature_dim)
            ],
            dim=-1,
        )  # (num_edges, 3*feature_dim)

        atom_updates = self.atom_update_gated_mlp(
            concat_features
        ) * self.initial_edge_linear_2(
            initial_edge_features
        )  # (num_edges, feature_dim)

        atomic_features = torch.scatter_add(
            atomic_features,
            dim=0,
            index=edge_index[0].unsqueeze(-1).expand(-1, self.feature_dim),
            src=atom_updates,
        )

        return atomic_features, edge_features


class ThreeBodyInteraction(nn.Module):
    """The three-body interaction layer."""

    def __init__(
        self,
        *,
        max_angular_l: int,
        max_radial_n: int,
        cutoff: float,
        three_body_cutoff: float,
        feature_dim: int,
    ):
        """Initialize the ThreeBodyInteraction class.

        Args:
            max_angular_l (int): The maximum angular momentum.
            max_radial_n (int): The maximum number of radial basis functions.
            cutoff (float): The cutoff radius.
            three_body_cutoff (float): The three-body cutoff radius.
            feature_dim (int): The feature dimension.
        """
        super().__init__()

        self.max_angular_l = max_angular_l
        self.max_radial_n = max_radial_n
        self.cutoff = cutoff
        self.three_body_cutoff = three_body_cutoff
        self.feature_dim = feature_dim
        self.angle_feature_dim = max_angular_l * max_radial_n

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
        atomic_features: torch.Tensor,
        edge_features: torch.Tensor,
        angle_features: torch.Tensor,
        three_body_indices_with_offset: torch.Tensor,
        edge_index: torch.Tensor,
        edge_dist: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for the three-body interaction layer.

        Args:
            atomic_features (torch.Tensor): Atomic features.
            edge_features (torch.Tensor): Edge features.
            angle_features (torch.Tensor): Angle features.
            three_body_indices_with_offset (torch.Tensor): Three-body indices.
            edge_index (torch.Tensor): Edge indices.
            edge_dist (torch.Tensor): Edge distances.

        Returns:
            torch.Tensor: The three-body interaction.
        """
        # apply atomwise MLP
        atomic_filter = self.atom_mlp(atomic_features)

        # Extract edge indices once to avoid repeated indexing
        edge_ij_indices = three_body_indices_with_offset[:, 0]  # Shape: [num_angles]
        edge_ik_indices = three_body_indices_with_offset[:, 1]  # Shape: [num_angles]

        # Get atomic filters for the central atoms (more efficient indexing)
        atomic_filter_k = atomic_filter[
            edge_index[1, edge_ik_indices]
        ]  # Shape: [num_angles, angle_feature_dim]

        # Compute envelope functions for both edges
        envelope_ij = envelope_polynomial(
            edge_dist[edge_ij_indices], self.three_body_cutoff
        ).unsqueeze(-1)  # Shape: [num_angles, 1]
        envelope_ik = envelope_polynomial(
            edge_dist[edge_ik_indices], self.three_body_cutoff
        ).unsqueeze(-1)  # Shape: [num_angles, 1]

        masks = atomic_filter_k * envelope_ij * envelope_ik

        # Vectorized accumulation of masked angle features
        edge_feature_ij_tilde = torch.zeros(
            [edge_index.shape[1], self.angle_feature_dim],
            device=edge_index.device,
        )

        # Apply masks to angle features (element-wise multiplication)
        masked_angle_features = (
            angle_features * masks
        )  # Shape: [num_angles, angle_feature_dim]

        # Use scatter_add to accumulate masked features for each edge
        edge_feature_ij_tilde = torch.scatter_add(
            edge_feature_ij_tilde,
            dim=0,
            index=edge_ij_indices.unsqueeze(-1).expand(-1, self.angle_feature_dim),
            src=masked_angle_features,
        )

        return edge_features + self.edge_gated_mlp(edge_feature_ij_tilde)
