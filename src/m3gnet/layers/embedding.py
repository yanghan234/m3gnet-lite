"""The embedding layers for atoms."""

import torch
from torch import nn


class AtomicEmbedding(nn.Module):
    """The atomic embedding layer.

    This layer maps atomic numbers to dense feature vectors.

    Args:
        max_num_elements (int): Maximum number of unique elements. Default is 108.
        feature_dim (int): Dimension of the atomic feature vectors. Default is 128.
    """

    def __init__(
        self,
        max_num_elements: int = 108,
        feature_dim: int = 128,
    ):
        """Initialize the AtomicEmbedding class."""
        super().__init__()
        self.max_num_elements = max_num_elements
        self.feature_dim = feature_dim
        self.embedding = nn.Embedding(max_num_elements + 1, feature_dim)
        # +1 because the atomic number starts from 1
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters of the embedding layer."""
        self.embedding.weight.data.normal_(0, 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the atomic embedding layer."""
        return self.embedding(x)
