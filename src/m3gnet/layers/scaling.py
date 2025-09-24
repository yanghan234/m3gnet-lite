"""Module for atomic energy scaling and shifting."""

import torch
from torch import nn


class AtomicScaling(nn.Module):
    """Element-wise scaling and shifting of atomic energies."""

    def __init__(
        self,
        *,
        num_elements: int = 108,
        trainable: bool = True,
        init_shift: float | None = None,
        init_scale: float | None = None,
    ):
        """Initialize the AtomicScaling layer.

        This layer applies element-wise scaling and shifting to atomic energies.

        Args:
            num_elements (int): Number of unique elements. Default is 108.
            trainable (bool):
                Whether the scaling and shifting parameters are trainable.
                Default is True.
            init_shift (float | None):
                Initial value for the shift parameters.
                If None, initialized to zeros. Default is None.
            init_scale (float | None):
                Initial value for the scale parameters.
                If None, initialized to ones. Default is None.
        """
        super().__init__()

        self.num_elements = num_elements
        self.trainable = trainable
        self.init_shift = init_shift or torch.zeros(
            num_elements + 1
        )  # +1 for padding index 0
        self.init_scale = init_scale or torch.ones(
            num_elements + 1
        )  # +1 for padding index 0

        if self.trainable:
            self.shift = nn.Parameter(self.init_shift)
            self.scale = nn.Parameter(self.init_scale)
        else:
            self.register_buffer("shift", self.init_shift)
            self.register_buffer("scale", self.init_scale)

    def normalize(
        self,
        *,
        atomic_numbers: torch.Tensor,
        atomic_energies: torch.Tensor,
    ):
        """Normalize atomic energies.

        Args:
            atomic_numbers (torch.Tensor):
                Tensor of shape (N,) containing atomic numbers.
            atomic_energies (torch.Tensor):
                Tensor of shape (N,) containing atomic energies.

        Returns:
            torch.Tensor: Normalized atomic energies of shape (N,).
        """
        shifts = self.shift[atomic_numbers]
        scales = self.scale[atomic_numbers]
        return (atomic_energies - shifts) / scales

    def denormalize(
        self,
        *,
        atomic_numbers: torch.Tensor,
        atomic_energies: torch.Tensor,
    ):
        """Denormalize atomic energies.

        Args:
            atomic_numbers (torch.Tensor):
                Tensor of shape (N,) containing atomic numbers.
            atomic_energies (torch.Tensor):
                Tensor of shape (N,) containing normalized atomic energies.

        Returns:
            torch.Tensor: Denormalized atomic energies of shape (N,).
        """
        shifts = self.shift[atomic_numbers]
        scales = self.scale[atomic_numbers]
        return atomic_energies * scales + shifts

    def forward(
        self,
        *,
        atomic_numbers: torch.Tensor,
        atomic_energies: torch.Tensor,
    ):
        """Forward pass of the AtomicScaling layer.

        This method normalizes the input atomic energies.

        Args:
            atomic_numbers (torch.Tensor):
                Tensor of shape (N,) containing atomic numbers.
            atomic_energies (torch.Tensor):
                Tensor of shape (N,) containing atomic energies.

        Returns:
            torch.Tensor: Normalized atomic energies of shape (N,).
        """
        return self.denormalize(
            atomic_numbers=atomic_numbers,
            atomic_energies=atomic_energies,
        )
