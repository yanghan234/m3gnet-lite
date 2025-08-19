from ase.io import read as ase_read
from loguru import logger
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from m3gnet.graph.converter import GraphConverter
from m3gnet.m3gnet import M3GNet

if __name__ == "__main__":
    # create a simple graph
    atoms_list = ase_read("data_samples/mpf-TP.xyz", index=":")
    converter = GraphConverter()
    data_list = [converter.convert_ase_atoms(atoms) for atoms in tqdm(atoms_list)]
    loader = DataLoader(data_list, batch_size=2, shuffle=False)

    # create a m3gnet model
    model = M3GNet()
    for data in loader:
        logger.info(data)
        logger.info(data[0])
        logger.info(data[1])
        # # output = model(data)
        # logger.info(output)
        break
