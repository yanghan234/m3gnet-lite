"""Lightning module for M3GNet model training and evaluation."""

import lightning
import torch
from torch import nn

from .m3gnet import M3GNet


class LightningM3GNet(lightning.LightningModule):
    """Lightning module for M3GNet model training and evaluation.

    Args:
        model (M3GNet): An instance of the M3GNet model.
        include_forces (bool): Whether to include forces in the loss calculation.
            Default is False.
        include_stresses (bool): Whether to include stresses in the loss calculation.
            Default is False.
        loss_forces_weight (float): Weight for the forces loss component.
            Default is 1.0.
        loss_stresses_weight (float): Weight for the stresses loss component.
            Default is 0.1.
        weight_decay (float): Weight decay for the optimizer.
            Default is 1e-2.
    """

    def __init__(
        self,
        model: M3GNet,
        include_forces: bool = False,
        include_stresses: bool = False,
        loss_forces_weight: float = 1.0,
        loss_stresses_weight: float = 0.1,
        weight_decay: float = 1e-2,
    ):
        """Initialize the LightningM3GNet module."""
        super().__init__()
        self.model = model
        self.include_forces = include_forces
        self.include_stresses = include_stresses
        self.loss_forces_weight = loss_forces_weight
        self.loss_stresses_weight = loss_stresses_weight
        self.weight_decay = weight_decay
        self.save_hyperparameters(ignore=["model"])

    def training_step(self, batch):
        """Training step."""
        num_atoms = batch.total_num_atoms.view(-1, 1)

        # Determine the inputs for the model based on whether stress is calculated
        model_input_batch = batch
        strain = None
        pos_for_grad = batch.pos.clone().requires_grad_()

        if self.include_stresses:
            actual_batch_size = num_atoms.shape[0]
            # Create a strain tensor
            strain = torch.zeros(
                actual_batch_size,
                3,
                3,
                device=batch.cell.device,
                dtype=batch.cell.dtype,
                requires_grad=True,
            )

            # Clone the batch to avoid modifying the original data from the dataloader
            strained_batch = batch.clone()

            # Apply strain to cell
            strained_cell = torch.matmul(
                batch.cell.reshape(-1, 3, 3),
                strain + torch.eye(3, device=batch.cell.device).unsqueeze(0),
            )
            strained_batch.cell = strained_cell.reshape(batch.cell.shape)
            volumes = torch.linalg.det(strained_batch.cell.reshape(-1, 3, 3))  # (B,)

            # Apply strain to positions
            strained_pos = torch.einsum(
                "bi, bij->bj",
                pos_for_grad,
                torch.repeat_interleave(strain, batch.total_num_atoms, dim=0)
                + torch.eye(3, device=batch.cell.device).unsqueeze(0),
            )
            strained_batch.pos = strained_pos.requires_grad_()

            model_input_batch = strained_batch
            pos_for_grad = strained_pos

        pred = self.model(model_input_batch)
        pred_energy_pa = pred.view(-1, 1) / num_atoms
        reference_energy_pa = batch.energy.view(-1, 1) / num_atoms
        loss = loss_e = nn.functional.l1_loss(pred_energy_pa, reference_energy_pa)
        self.log("train/e_loss", loss_e)
        self.log("train/e_loss_weighted", loss_e)

        if not self.include_forces and not self.include_stresses:
            pass
        elif self.include_forces and not self.include_stresses:
            forces = -torch.autograd.grad(pred.sum(), pos_for_grad, create_graph=True)[
                0
            ]
            loss_f = nn.functional.mse_loss(forces, batch.forces)
            loss = loss + loss_f * self.loss_forces_weight
            self.log("train/f_loss", loss_f)
            self.log("train/f_loss_weighted", loss_f * self.loss_forces_weight)
        elif not self.include_forces and self.include_stresses:
            # Compute stressesm, shape (B, 3, 3)
            stresses = torch.autograd.grad(pred.sum(), strain, create_graph=True)[0]
            stresses = stresses / volumes.view(-1, 1, 1)
            loss_s = nn.functional.mse_loss(stresses, batch.stress.view(-1, 3, 3))
            loss = loss + loss_s * self.loss_stresses_weight
            self.log("train/s_loss", loss_s)
            self.log("train/s_loss_weighted", loss_s * self.loss_stresses_weight)
        else:  # include both forces and stresses
            forces, stresses = torch.autograd.grad(
                pred.sum(), [pos_for_grad, strain], create_graph=True
            )
            forces = -forces
            loss_f = nn.functional.mse_loss(forces, batch.forces)
            stresses = stresses / volumes.view(-1, 1, 1)
            loss_s = nn.functional.mse_loss(stresses, batch.stress.view(-1, 3, 3))
            loss = (
                loss
                + loss_f * self.loss_forces_weight
                + loss_s * self.loss_stresses_weight
            )
            self.log("train/f_loss", loss_f)
            self.log("train/f_loss_weighted", loss_f * self.loss_forces_weight)
            self.log("train/s_loss", loss_s)
            self.log("train/s_loss_weighted", loss_s * self.loss_stresses_weight)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch):
        """Validation step."""
        num_atoms = batch.total_num_atoms.view(-1, 1)

        # Determine the inputs for the model based on whether stress is calculated
        model_input_batch = batch
        strain = None
        with torch.enable_grad():
            pos_for_grad = batch.pos.clone().requires_grad_()

            if self.include_stresses:
                actual_batch_size = num_atoms.shape[0]
                # Create a strain tensor
                strain = torch.zeros(
                    actual_batch_size,
                    3,
                    3,
                    device=batch.cell.device,
                    dtype=batch.cell.dtype,
                    requires_grad=True,
                )

                # Clone the batch to avoid modifying
                # the original data from the dataloader
                strained_batch = batch.clone()

                # Apply strain to cell
                strained_cell = torch.matmul(
                    batch.cell.reshape(-1, 3, 3),
                    strain + torch.eye(3, device=batch.cell.device).unsqueeze(0),
                )
                strained_batch.cell = strained_cell.reshape(batch.cell.shape)
                volumes = torch.linalg.det(
                    strained_batch.cell.reshape(-1, 3, 3)
                )  # (B,)

                # Apply strain to positions
                strained_pos = torch.einsum(
                    "bi, bij->bj",
                    pos_for_grad,
                    torch.repeat_interleave(strain, batch.total_num_atoms, dim=0)
                    + torch.eye(3, device=batch.cell.device).unsqueeze(0),
                )
                strained_batch.pos = strained_pos.requires_grad_()

                model_input_batch = strained_batch
                pos_for_grad = strained_pos

            pred = self.model(model_input_batch)
            pred_energy_pa = pred.view(-1, 1) / num_atoms
            reference_energy_pa = batch.energy.view(-1, 1) / num_atoms
            loss = loss_e = nn.functional.l1_loss(pred_energy_pa, reference_energy_pa)
            self.log("train/e_loss", loss_e)
            self.log("train/e_loss_weighted", loss_e)

            if not self.include_forces and not self.include_stresses:
                pass
            elif self.include_forces and not self.include_stresses:
                forces = -torch.autograd.grad(
                    pred.sum(), pos_for_grad, create_graph=True
                )[0]
                loss_f = nn.functional.mse_loss(forces, batch.forces)
                loss = loss + loss_f * self.loss_forces_weight
                self.log("train/f_loss", loss_f)
                self.log("train/f_loss_weighted", loss_f * self.loss_forces_weight)
            elif not self.include_forces and self.include_stresses:
                # Compute stressesm, shape (B, 3, 3)
                stresses = torch.autograd.grad(pred.sum(), strain, create_graph=True)[0]
                stresses = stresses / volumes.view(-1, 1, 1)
                loss_s = nn.functional.mse_loss(stresses, batch.stress.view(-1, 3, 3))
                loss = loss + loss_s * self.loss_stresses_weight
                self.log("train/s_loss", loss_s)
                self.log("train/s_loss_weighted", loss_s * self.loss_stresses_weight)
            else:  # include both forces and stresses
                forces, stresses = torch.autograd.grad(
                    pred.sum(), [pos_for_grad, strain], create_graph=True
                )
                forces = -forces
                loss_f = nn.functional.mse_loss(forces, batch.forces)
                stresses = stresses / volumes.view(-1, 1, 1)
                loss_s = nn.functional.mse_loss(stresses, batch.stress.view(-1, 3, 3))
                loss = (
                    loss
                    + loss_f * self.loss_forces_weight
                    + loss_s * self.loss_stresses_weight
                )
                self.log("train/f_loss", loss_f)
                self.log("train/f_loss_weighted", loss_f * self.loss_forces_weight)
                self.log("train/s_loss", loss_s)
                self.log("train/s_loss_weighted", loss_s * self.loss_stresses_weight)
            self.log("train/loss", loss)
        return loss

    def configure_optimizers(self):
        """Configure optimizers."""
        return torch.optim.AdamW(
            self.model.parameters(), lr=1e-3, weight_decay=self.weight_decay
        )
