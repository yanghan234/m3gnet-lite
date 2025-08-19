import torch
from torch import nn


class AtomicEmbedding(nn.Module):
    def __init__(
        self,
        max_num_elements: int = 108,
        feature_dim: int = 128,
        *,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.max_num_elements = max_num_elements
        self.feature_dim = feature_dim
        self.device = device
        self.embedding = nn.Embedding(max_num_elements + 1, feature_dim)
        # +1 because the atomic number starts from 1
        self.reset_parameters()

        # move model to device
        self.to(device)

    def reset_parameters(self) -> None:
        self.embedding.weight.data.normal_(0, 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)
