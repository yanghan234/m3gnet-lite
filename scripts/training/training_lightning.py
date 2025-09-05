"""Script to train M3GNet model using PyTorch Lightning."""

import lightning as lightning
from torch.utils.data import random_split

from m3gnet import LightningM3GNet

if __name__ == "__main__":
    from torch_geometric.loader import DataLoader

    from m3gnet import M3GNet
    from m3gnet.datasets import MPF2021Dataset
    from m3gnet.utils import ensure_reproducibility

    ensure_reproducibility(42)

    mpf2021_dataset = MPF2021Dataset()
    train_size = int(0.9 * len(mpf2021_dataset))
    val_size = int(0.05 * len(mpf2021_dataset))
    test_size = len(mpf2021_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        mpf2021_dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model = M3GNet()
    lightning_model = LightningM3GNet(
        model=model, include_forces=True, include_stresses=True
    )

    trainer = lightning.Trainer(max_epochs=10, accelerator="auto")
    trainer.fit(lightning_model, train_loader, val_loader)
