import numpy as np
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


class SmoothBesselBasis(nn.Module):
    def __init__(
        self,
        cutoff: float,
        max_radial_n: int = 3,
        device: str | torch.device = "cpu",
    ):
        super().__init__()

        self.cutoff = cutoff
        self.max_radial_n = max_radial_n
        self.device = device

        # generate temporary parameters
        en = torch.zeros(max_radial_n + 1, dtype=torch.float32, device=device)
        for n in range(max_radial_n + 1):
            en[n] = n**2 * (n + 2) ** 2 / (4 * (n + 1) ** 4 + 1)
        self.register_buffer("en", en)

        dn = torch.arange(max_radial_n + 1, dtype=torch.float32, device=device)
        for n in range(max_radial_n + 1):
            dn[n] = 1.0 - en[n] / dn[n - 1] if n > 0 else 1.0
        self.register_buffer("dn", dn)

        n_plus_1_factor = [
            (n + 1) * np.pi / self.cutoff for n in range(max_radial_n + 1)
        ]
        n_plus_2_factor = [
            (n + 2) * np.pi / self.cutoff for n in range(max_radial_n + 1)
        ]
        self.register_buffer(
            "n_plus_1_factor",
            torch.tensor(n_plus_1_factor, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "n_plus_2_factor",
            torch.tensor(n_plus_2_factor, dtype=torch.float32, device=device),
        )

        fn_factor = [
            (-1) ** n
            * np.sqrt(2)
            * np.pi
            / self.cutoff**1.5
            * (n + 1)
            * (n + 2)
            / np.sqrt((n + 1) ** 2 + (n + 2) ** 2)
            for n in range(max_radial_n + 1)
        ]
        self.register_buffer(
            "fn_factor", torch.tensor(fn_factor, dtype=torch.float32, device=device)
        )

        # generate parameters
        self.to(device)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate the smooth bessel basis functions at a give coordinate x.

        Args:
            x: (num_edges,)

        Returns:
            (num_edges, max_radial_n)
        """

        bessel_basis = torch.zeros(
            x.shape[0], self.max_radial_n + 1, device=self.device
        )

        for n in range(self.max_radial_n + 1):
            r_with_plus1_factor = x * self.n_plus_1_factor[n]
            r_with_plus2_factor = x * self.n_plus_2_factor[n]

            # please note, in numpy and torch, sinc(x) = sin(pi * x) / (pi * x)
            # but I decide to use the implementation in numpy/torch,
            # because it is more stable especially when x is close to 0
            fn = self.fn_factor[n] * (
                torch.sinc(r_with_plus1_factor / np.pi)
                + torch.sinc(r_with_plus2_factor / np.pi)
            )
            if n == 0:
                bessel_basis[:, n] = fn
            else:
                bessel_basis[:, n] = (
                    1
                    / torch.sqrt(self.dn[n])
                    * (
                        fn
                        + torch.sqrt(self.en[n] / self.dn[n - 1])
                        * bessel_basis[:, n - 1]
                    )
                )
        return bessel_basis
