"""Lightning module for M3GNet model training and evaluation."""

import lightning
import torch
from ase.units import GPa
from torch import nn
from typing import Literal
from loguru import logger

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
        learning_rate (float): Learning rate for the optimizer.
            Default is 1e-3.
        weight_decay (float): Weight decay for the optimizer.
            Default is 1e-2.
        loss (Literal["l1", "mse", "huber"]): Loss function type.
            Default is "huber".
    """

    def __init__(
        self,
        model: M3GNet,
        include_forces: bool = False,
        include_stresses: bool = False,
        loss_forces_weight: float = 1.0,
        loss_stresses_weight: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-2,
        scheduler: Literal["StepLR", "OneCycleLR"] = "OneCycleLR",
        step_scheduler_gamma: float = 0.9,
        step_scheduler_step_size: int = 10,
        onecycle_scheduler_pct_start: float = 0.1,
        onecycle_scheduler_anneal_strategy: str = 'cos',
        loss: Literal["l1", "mse", "huber"] = "huber",
    ):
        """Initialize the LightningM3GNet module."""
        super().__init__()
        self.model = model
        self.include_forces = include_forces
        self.include_stresses = include_stresses
        self.loss_forces_weight = loss_forces_weight
        self.loss_stresses_weight = loss_stresses_weight
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.step_scheduler_gamma = step_scheduler_gamma
        self.step_scheduler_step_size = step_scheduler_step_size
        self.onecycle_scheduler_pct_start = onecycle_scheduler_pct_start
        self.onecycle_scheduler_anneal_strategy = onecycle_scheduler_anneal_strategy
        if self.scheduler == "StepLR":
            logger.warning(
                f"Using StepLR scheduler, with gamma={self.step_scheduler_gamma} "
                f"and step_size={self.step_scheduler_step_size}. "
                "OneCycleLR related arguments will be ignored."
            )
        elif self.scheduler == "OneCycleLR":
            logger.warning(
                "Using OneCycleLR scheduler, "
                f"with pct_start={self.onecycle_scheduler_pct_start} "
                f"and anneal_strategy={self.onecycle_scheduler_anneal_strategy}. "
                "StepLR related arguments will be ignored."
            )
        else:
            raise ValueError("Scheduler must be one of 'StepLR' or 'OneCycleLR'.")
        self.loss = loss
        if loss.lower() not in ["l1", "mse", "huber"]:
            raise ValueError("Loss must be one of 'l1', 'mse', or 'huber'.")
        if self.loss == "l1":
            self.energy_loss_fn = nn.L1Loss()
            self.forces_loss_fn = nn.L1Loss()
            self.stresses_loss_fn = nn.L1Loss()
        elif self.loss == "mse":
            self.energy_loss_fn = nn.MSELoss()
            self.forces_loss_fn = nn.MSELoss()
            self.stresses_loss_fn = nn.MSELoss()
        else:  # huber
            self.energy_loss_fn = nn.HuberLoss(delta=0.01)
            self.forces_loss_fn = nn.HuberLoss(delta=0.01)
            self.stresses_loss_fn = nn.HuberLoss(delta=0.01)
        self.save_hyperparameters(ignore=["model"])

    def training_step(self, batch):
        """Training step."""
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_epoch=True, on_step=True)
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
        loss = loss_e = self.energy_loss_fn(pred_energy_pa, reference_energy_pa)
        self.log("train_e_loss", loss_e)
        self.log("train_e_loss_weighted", loss_e)
        self.log("train_e_mse_loss", nn.functional.mse_loss(pred_energy_pa, reference_energy_pa))
        self.log("train_e_mae_loss", nn.functional.l1_loss(pred_energy_pa, reference_energy_pa))

        if not self.include_forces and not self.include_stresses:
            pass
        elif self.include_forces and not self.include_stresses:
            forces = -torch.autograd.grad(pred.sum(), pos_for_grad, create_graph=True)[
                0
            ]
            loss_f = self.forces_loss_fn(forces, batch.forces)
            # loss_f = nn.functional.huber_loss(forces, batch.forces)
            loss = loss + loss_f * self.loss_forces_weight
            self.log("train_f_loss", loss_f)
            self.log("train_f_loss_weighted", loss_f * self.loss_forces_weight)
            self.log("train_f_mse_loss", nn.functional.mse_loss(forces, batch.forces))
            self.log("train_f_mae_loss", nn.functional.l1_loss(forces, batch.forces))
        elif not self.include_forces and self.include_stresses:
            # Compute stressesm, shape (B, 3, 3)
            stresses = torch.autograd.grad(pred.sum(), strain, create_graph=True)[0]
            stresses = stresses / volumes.view(-1, 1, 1) / GPa
            loss_s = self.stresses_loss_fn(stresses, batch.stress.view(-1, 3, 3))
            loss = loss + loss_s * self.loss_stresses_weight
            self.log("train_s_loss", loss_s)
            self.log("train_s_loss_weighted", loss_s * self.loss_stresses_weight)
            self.log("train_s_mse_loss", nn.functional.mse_loss(stresses, batch.stress.view(-1, 3, 3)))
            self.log("train_s_mae_loss", nn.functional.l1_loss(stresses, batch.stress.view(-1, 3, 3)))
        else:  # include both forces and stresses
            forces, stresses = torch.autograd.grad(
                pred.sum(), [pos_for_grad, strain], create_graph=True
            )
            forces = -forces
            loss_f = nn.functional.huber_loss(forces, batch.forces)
            stresses = stresses / volumes.view(-1, 1, 1) / GPa
            loss_s = self.stresses_loss_fn(stresses, batch.stress.view(-1, 3, 3))
            loss = (
                loss
                + loss_f * self.loss_forces_weight
                + loss_s * self.loss_stresses_weight
            )
            self.log("train_f_loss", loss_f)
            self.log("train_f_loss_weighted", loss_f * self.loss_forces_weight)
            self.log("train_f_mse_loss", nn.functional.mse_loss(forces, batch.forces))
            self.log("train_f_mae_loss", nn.functional.l1_loss(forces, batch.forces))
            self.log("train_s_loss", loss_s)
            self.log("train_s_loss_weighted", loss_s * self.loss_stresses_weight)
            self.log("train_s_mse_loss", nn.functional.mse_loss(stresses, batch.stress.view(-1, 3, 3)))
            self.log("train_s_mae_loss", nn.functional.l1_loss(stresses, batch.stress.view(-1, 3, 3)))
        self.log("train_loss", loss)
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
            # loss = loss_e = nn.functional.l1_loss(pred_energy_pa, reference_energy_pa)
            loss = loss_e = self.energy_loss_fn(pred_energy_pa, reference_energy_pa)
            self.log("val_e_loss", loss_e)
            self.log("val_e_loss_weighted", loss_e)
            self.log("val_e_mse_loss", nn.functional.mse_loss(pred_energy_pa, reference_energy_pa))
            self.log("val_e_mae_loss", nn.functional.l1_loss(pred_energy_pa, reference_energy_pa))

            if not self.include_forces and not self.include_stresses:
                pass
            elif self.include_forces and not self.include_stresses:
                forces = -torch.autograd.grad(
                    pred.sum(), pos_for_grad, create_graph=True
                )[0]
                loss_f = nn.functional.huber_loss(forces, batch.forces)
                loss = loss + loss_f * self.loss_forces_weight
                self.log("val_f_loss", loss_f)
                self.log("val_f_loss_weighted", loss_f * self.loss_forces_weight)
                self.log("val_f_mse_loss", nn.functional.mse_loss(forces, batch.forces))
                self.log("val_f_mae_loss", nn.functional.l1_loss(forces, batch.forces))
            elif not self.include_forces and self.include_stresses:
                # Compute stressesm, shape (B, 3, 3)
                stresses = torch.autograd.grad(pred.sum(), strain, create_graph=True)[0]
                stresses = stresses / volumes.view(-1, 1, 1) / GPa
                loss_s = nn.functional.huber_loss(stresses, batch.stress.view(-1, 3, 3))
                loss = loss + loss_s * self.loss_stresses_weight
                self.log("val_s_loss", loss_s)
                self.log("val_s_loss_weighted", loss_s * self.loss_stresses_weight)
                self.log("val_s_mse_loss", nn.functional.mse_loss(stresses, batch.stress.view(-1, 3, 3)))
                self.log("val_s_mae_loss", nn.functional.l1_loss(stresses, batch.stress.view(-1, 3, 3)))
            else:  # include both forces and stresses
                forces, stresses = torch.autograd.grad(
                    pred.sum(), [pos_for_grad, strain], create_graph=True
                )
                forces = -forces
                loss_f = nn.functional.huber_loss(forces, batch.forces)
                stresses = stresses / volumes.view(-1, 1, 1) / GPa
                loss_s = nn.functional.huber_loss(stresses, batch.stress.view(-1, 3, 3))
                loss = (
                    loss
                    + loss_f * self.loss_forces_weight
                    + loss_s * self.loss_stresses_weight
                )
                self.log("val_f_loss", loss_f)
                self.log("val_f_loss_weighted", loss_f * self.loss_forces_weight)
                self.log("val_f_mse_loss", nn.functional.mse_loss(forces, batch.forces))
                self.log("val_f_mae_loss", nn.functional.l1_loss(forces, batch.forces))
                self.log("val_s_loss", loss_s)
                self.log("val_s_loss_weighted", loss_s * self.loss_stresses_weight)
                self.log("val_s_mse_loss", nn.functional.mse_loss(stresses, batch.stress.view(-1, 3, 3)))
                self.log("val_s_mae_loss", nn.functional.l1_loss(stresses, batch.stress.view(-1, 3, 3)))
            self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        """Configure optimizers."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        if self.scheduler == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.step_scheduler_step_size,
                gamma=self.step_scheduler_gamma
            )
        else:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, 
                max_lr=self.learning_rate,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=self.onecycle_scheduler_pct_start,
                anneal_strategy=self.onecycle_scheduler_anneal_strategy
            )
        return (
            [optimizer],
            [
                {
                    "scheduler": scheduler, 
                    "interval": "epoch" if self.scheduler == "StepLR" else "step",
                    "frequency": 1,
                    "name": self.scheduler.lower()
                }
            ]
        )
        