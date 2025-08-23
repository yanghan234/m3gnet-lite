"""A quick run script to test the M3GNet model."""

import random

import numpy as np
import torch
from ase.io import read as ase_read
from loguru import logger
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from m3gnet.graph.converter import GraphConverter
from m3gnet.m3gnet import M3GNet

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # create a simple graph
    atoms_list = ase_read("data_samples/mpf-TP.xyz", index=":10")
    converter = GraphConverter()
    data_list = [converter.convert_ase_atoms(atoms) for atoms in tqdm(atoms_list)]
    loader = DataLoader(data_list, batch_size=2, shuffle=False)

    # c = bulk("C", "diamond", a=3.567)
    # c.calc = EMT()
    # data_list = [
    #     GraphConverter().convert_ase_atoms(c),
    #     GraphConverter().convert_ase_atoms(c),
    # ]
    # loader = DataLoader(data_list, batch_size=2, shuffle=False)

    # create a m3gnet model
    model = M3GNet()

    for _idx, data in enumerate(loader):
        output = model(data)
        # Test that forces can be computed
        forces = torch.autograd.grad(output.sum(), data.pos, create_graph=False)[0]
        logger.info(f"Forces computed successfully! Shape: {forces.shape}")
        logger.info(f"Forces max magnitude: {torch.max(torch.norm(forces, dim=1)):.6f}")
        # logger.info(f"Forces:\n {forces}")
        break
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total number of parameters: {total_params}")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total number of trainable parameters: {trainable_params}")
