"""test training script."""

import torch
from loguru import logger
from torch.nn import L1Loss
from torch.optim import AdamW
from torch_geometric.loader import DataLoader

from m3gnet import M3GNet
from m3gnet.datasets import MPF2021Dataset
from m3gnet.utils import ensure_reproducibility


def test_device():
    """Test device."""
    ensure_reproducibility(random_seed=42)

    ## Choose device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    # torch.set_default_device(device)

    ## define the model
    model = M3GNet().to(device)
    dataset = MPF2021Dataset()
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    for batch in dataloader:
        batch = batch.to(device)
        output = model(batch)
        print(output)
        break


def main():
    """Main function."""
    num_epochs = 10
    learning_rate = 0.001
    random_seed = 42
    batch_size = 32

    ensure_reproducibility(random_seed)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    """Main function."""
    model = M3GNet().to(device)

    # define the dataset
    dataset = MPF2021Dataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # define loss function
    energy_loss_fn = L1Loss()

    # define the optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    total_number_of_steps = len(dataset) // batch_size
    logger.info(f"Total number of steps: {total_number_of_steps}")

    # train the model
    for epoch_idx in range(num_epochs):
        for step_idx, data_batch in enumerate(dataloader):
            data_batch = data_batch.to(device)
            total_num_atoms = data_batch.total_num_atoms
            data_batch.pos.requires_grad_()
            pred = model(data_batch)
            energy_loss = energy_loss_fn(
                pred / total_num_atoms, data_batch.energy / total_num_atoms
            )

            # backpropagation
            optimizer.zero_grad()
            energy_loss.backward()
            optimizer.step()
            logger.info(
                f"Epoch {epoch_idx:04d}/{num_epochs:04d}, "
                f"Step {step_idx:08d}/{total_number_of_steps:08d}, "
                f"Energy Loss: {energy_loss.item():8.4e}"
            )
            # break


if __name__ == "__main__":
    main()
    # test_device()
