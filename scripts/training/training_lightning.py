"""Script to train M3GNet model using PyTorch Lightning."""

import lightning as lightning
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import random_split
from loguru import logger

from m3gnet import LightningM3GNet
import argparse

def main():
    parser = argparse.ArgumentParser(description="Train M3GNet model using PyTorch Lightning")
    parser.add_argument("--project", type=str, default="m3gnet", help="WandB project name")
    parser.add_argument("--log-model", type=str, default="all", help="WandB log model option")
    parser.add_argument("--max-epochs", type=int, default=200, help="Maximum number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training and validation")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for the optimizer")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for the optimizer")
    parser.add_argument("--gradient-clip-val", type=float, default=0.1, help="Gradient clipping value")
    parser.add_argument("--gradient-clip-algorithm", type=str, default="norm", help="Gradient clipping algorithm (e.g., 'norm', 'value')")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--accelerator", type=str, default="auto", help="Accelerator to use (e.g., 'cpu', 'gpu', 'mps', 'auto')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility, default is 42.")
    parser.add_argument("--include-forces", action=argparse.BooleanOptionalAction, help="Whether to include forces in the training", default=True)
    parser.add_argument("--include-stresses", action=argparse.BooleanOptionalAction, help="Whether to include stresses in the training", default=True)
    parser.add_argument("--scheduler", type=str, default="OneCycleLR", choices=["StepLR", "OneCycleLR"], help="Type of learning rate scheduler to use")
    parser.add_argument("--step-scheduler-gamma", type=float, default=0.9, help="Gamma value for StepLR scheduler")
    parser.add_argument("--step-scheduler-step-size", type=int, default=10, help="Step size in terms of epochs for StepLR scheduler")
    parser.add_argument("--onecycle-scheduler-pct-start", type=float, default=0.1, help="Percentage of the cycle spent increasing the learning rate for OneCycleLR scheduler")
    parser.add_argument("--onecycle-scheduler-anneal-strategy", type=str, default="cos", choices=["cos", "linear"], help="Annealing strategy for OneCycleLR scheduler")
    args = parser.parse_args()

    logger.info(f"Training configuration: {args}")

    from torch_geometric.loader import DataLoader
    from m3gnet import M3GNet
    from m3gnet.datasets import MPF2021Dataset
    from m3gnet.utils import ensure_reproducibility

    ensure_reproducibility(args.seed)

    mpf2021_dataset = MPF2021Dataset()
    train_size = int(0.9 * len(mpf2021_dataset))
    val_size = int(0.05 * len(mpf2021_dataset))
    test_size = len(mpf2021_dataset) - train_size - val_size
    train_dataset, val_dataset, _ = random_split(
        mpf2021_dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = M3GNet()
    lightning_model = LightningM3GNet(
        model=model,
        include_forces=args.include_forces,
        include_stresses=args.include_stresses,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler,
        step_scheduler_gamma=args.step_scheduler_gamma,
        step_scheduler_step_size=args.step_scheduler_step_size,
        onecycle_scheduler_pct_start=args.onecycle_scheduler_pct_start,
        onecycle_scheduler_anneal_strategy=args.onecycle_scheduler_anneal_strategy,
    )

    run_name = f"m3gnet_lr{args.learning_rate}_wd{args.weight_decay}_bs{args.batch_size}_epochs{args.max_epochs}_seed{args.seed}"

    wandb_logger = WandbLogger(project=args.project, log_model=args.log_model, name=run_name)
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{run_name}",
        filename="epoch-{epoch:04d}_val-loss{val_loss:8.4e}",
        monitor="val/loss",
        mode="min",
        save_top_k=5,
        save_last=True,
        every_n_epochs=1,
    )
    
    trainer = lightning.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        logger=wandb_logger,
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm=args.gradient_clip_algorithm,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(lightning_model, train_loader, val_loader)


if __name__ == "__main__":
    main()