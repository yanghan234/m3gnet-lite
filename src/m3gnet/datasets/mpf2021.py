"""MPF2021 Dataset."""

import json
import os
import pickle
from collections.abc import Callable

import lmdb
import numpy as np
import torch
from ase.data import atomic_numbers as ase_atomic_numbers
from loguru import logger
from torch_geometric.data import Dataset, download_url
from tqdm import tqdm

from m3gnet.graph import GraphConverter


class MPF2021Dataset(Dataset):
    """MPF2021 Dataset."""

    urls = [
        "https://figshare.com/ndownloader/files/37587100",
        "https://figshare.com/ndownloader/files/37587103",
    ]
    expected_number_of_trajectories = 62_783
    expected_number_of_structures = 187_687

    MAX_SHARD_SIZE_BYTES = 5 * 1024**3  # 5 GB

    def __init__(
        self,
        root: str = os.path.join(
            os.path.expanduser("~"), ".cache", "datasets", "mpf2021"
        ),
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
        graph_converter: GraphConverter = None,
    ):
        """MPF2021 Dataset."""
        self.graph_converter = graph_converter or GraphConverter()

        first_shard_path = os.path.join(root, "processed", "shard_0")
        if os.path.exists(first_shard_path):
            self.db_shards = self._load_shards_metadata(first_shard_path)
            self.num_samples = (
                self.db_shards[-1]["cumulative_entries"] if self.db_shards else 0
            )
        else:
            self.db_shards = []
            self.num_samples = 0
        self._db_connections = {}

        super().__init__(root, transform, pre_transform, pre_filter)

        # Reload metadata in case processing just finished
        if not self.db_shards and os.path.exists(first_shard_path):
            self.db_shards = self._load_shards_metadata(first_shard_path)
            self.num_samples = (
                self.db_shards[-1]["cumulative_entries"] if self.db_shards else 0
            )

    def _load_shards_metadata(self, first_shard_path: str) -> list[dict]:
        try:
            with lmdb.open(first_shard_path, readonly=True) as env, env.begin() as txn:
                metadata_bytes = txn.get(b"__metadata__")
                if metadata_bytes is None:
                    raise ValueError("Shard metadata not found in the database.")
                return json.loads(metadata_bytes.decode())
        except (lmdb.Error, FileNotFoundError) as e:
            logger.error(f"Failed to load shard metadata: {e}")
            return []

    @property
    def raw_file_names(self):
        """Get the names of the raw files."""
        return ["MPF.2021.2.8_block_0.pkl", "MPF.2021.2.8_block_1.pkl"]

    @property
    def processed_file_names(self):
        """Get the names of the processed files."""
        return [os.path.join(f"shard_{i}", "data.mdb") for i in range(8)]

    def download(self):
        """Download the raw data files."""
        download_url(self.urls[0], self.raw_dir, filename=self.raw_file_names[0])
        download_url(self.urls[1], self.raw_dir, filename=self.raw_file_names[1])

    def len(self):
        """Return the number of samples in the dataset."""
        return self.num_samples

    def process(
        self,
        chunk_size: int = 1000,
    ):
        """Process the raw data files and store them in LMDB shards."""
        with open(self.raw_paths[0], "rb") as f:
            raw_data = pickle.load(f)  # noqa: S301

        with open(self.raw_paths[1], "rb") as f:
            raw_data.update(pickle.load(f))  # noqa: S301

        if len(raw_data) != self.expected_number_of_trajectories:
            raise ValueError(
                f"Expected {self.expected_number_of_trajectories} trajectories, "
                f"but got {len(raw_data)}."
            )

        flatten_raw_data = []
        for mp_id, values in raw_data.items():
            for idx, structure in enumerate(values["structure"]):
                flatten_raw_data.append(
                    {
                        "mp_id": mp_id,
                        "snapshot_id": values["id"][idx],
                        "structure": structure,
                        "energy": values["energy"][idx],
                        "force": values["force"][idx],
                        "stress": values["stress"][idx],
                    }
                )

        number_of_structures = len(flatten_raw_data)
        if number_of_structures != self.expected_number_of_structures:
            raise ValueError(
                f"Expected {self.expected_number_of_structures} structures, "
                f"but got {number_of_structures}."
            )

        # sharding information
        shard_info = self.db_shards if self.db_shards else []
        start_idx = self.db_shards[-1]["cumulative_entries"] if self.db_shards else 0
        shard_idx = self.db_shards[-1]["shard_idx"] + 1 if self.db_shards else 0
        current_db_path = os.path.join(self.processed_dir, f"shard_{shard_idx}")
        env = lmdb.open(current_db_path, map_size=self.MAX_SHARD_SIZE_BYTES)
        txn = env.begin(write=True)
        current_shard_count = 0
        first_shard_path = os.path.join(self.processed_dir, "shard_0")

        # chunk
        chunk = {}

        for global_idx, values in enumerate(
            tqdm(flatten_raw_data, total=number_of_structures)
        ):
            if global_idx < start_idx:
                continue
            snapshot_id = values["snapshot_id"]
            structure = values["structure"]
            lattice = np.array(structure._lattice._matrix)  # noqa: SLF001
            chemical_symbols = [
                site._species.chemical_system  # noqa: SLF001
                for site in structure._sites  # noqa: SLF001
            ]
            cart_coords = np.array([site._coords for site in structure._sites])  # noqa: SLF001
            atomic_numbers = np.array(
                [ase_atomic_numbers[symbol] for symbol in chemical_symbols]
            )

            # convert to graph
            data = self.graph_converter.convert(
                pos=cart_coords, cell=lattice, atomic_numbers=atomic_numbers
            )
            data.mp_id = values["mp_id"]
            data.snapshot_id = snapshot_id
            data.energy = torch.tensor(values["energy"], dtype=torch.float32)
            data.forces = torch.tensor(values["force"], dtype=torch.float32)
            data.stress = torch.tensor(
                np.array(values["stress"]) * (-0.1), dtype=torch.float32
            )

            chunk[f"{global_idx}".encode()] = pickle.dumps(data)

            if len(chunk) < chunk_size:
                continue

            # try to put the chunk into the current shard
            try:
                for key, data in chunk.items():
                    txn.put(key, data)
                txn.commit()
                txn = env.begin(write=True)
                current_shard_count += len(chunk)
                chunk = {}
            except lmdb.MapFullError:
                # abort the current transaction
                txn.abort()
                env.close()

                # update shard metadata
                shard_info.append(
                    {
                        "shard_idx": shard_idx,
                        "shard_name": f"shard_{shard_idx}",
                        "cumulative_entries": shard_info[-1]["cumulative_entries"]
                        + current_shard_count
                        if shard_info
                        else current_shard_count,
                        "count": current_shard_count,
                    }
                )

                metadata_str = json.dumps(shard_info, indent=2)

                with (
                    lmdb.open(
                        first_shard_path, map_size=self.MAX_SHARD_SIZE_BYTES
                    ) as env0,
                    env0.begin(write=True) as txn0,
                ):
                    txn0.put(b"__metadata__", metadata_str.encode())
                # create new shard
                shard_idx += 1
                current_db_path = os.path.join(self.processed_dir, f"shard_{shard_idx}")
                env = lmdb.open(current_db_path, map_size=self.MAX_SHARD_SIZE_BYTES)
                txn = env.begin(write=True)
                current_shard_count = 0

                # retry putting the chunk
                for key, data in chunk.items():
                    txn.put(key, data)
                txn.commit()
                txn = env.begin(write=True)
                current_shard_count += len(chunk)
                chunk = {}

        # Put any remaining data in the chunk
        for key, data in chunk.items():
            txn.put(key, data)
        current_shard_count += len(chunk)
        chunk = {}

        # Final commit for the last shard
        txn.commit()
        env.close()

        shard_info.append(
            {
                "shard_idx": shard_idx,
                "shard_name": f"shard_{shard_idx}",
                "cumulative_entries": shard_info[-1]["cumulative_entries"]
                + current_shard_count
                if shard_info
                else current_shard_count,
                "count": current_shard_count,
            }
        )

        metadata_str = json.dumps(shard_info, indent=2)

        first_shard_path = os.path.join(self.processed_dir, "shard_0")
        with (
            lmdb.open(first_shard_path, map_size=self.MAX_SHARD_SIZE_BYTES) as env,
            env.begin(write=True) as txn,
        ):
            txn.put(b"__metadata__", metadata_str.encode())

    def get(self, idx: int):
        """Get the data object at index `idx`. TODO: support sharding."""
        if idx < 0 or idx >= self.num_samples:
            raise IndexError("Index out of bounds.")

        # Determine which shard the index belongs to
        shard_idx = 0
        while (
            shard_idx < len(self.db_shards)
            and idx >= self.db_shards[shard_idx]["cumulative_entries"]
        ):
            shard_idx += 1

        if shard_idx >= len(self.db_shards):
            raise IndexError("Shard index out of bounds.")

        shard_name = self.db_shards[shard_idx]["shard_name"]
        target_shard_path = os.path.join(self.processed_dir, shard_name)
        if shard_name not in self._db_connections:
            self._db_connections[shard_name] = lmdb.open(
                target_shard_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
        env = self._db_connections[shard_name]

        with env.begin() as txn:
            # Calculate the local index within the shard
            data_bytes = txn.get(f"{idx}".encode())
            if data_bytes is None:
                raise KeyError(f"Data for index {idx} not found in shard {shard_name}.")
            return pickle.loads(data_bytes)  # noqa: S301

    def __del__(self):
        """Close all open LMDB connections when the dataset object is destroyed."""
        for env in getattr(self, "_db_connections", {}).values():
            env.close()


if __name__ == "__main__":
    mpf_dataset = MPF2021Dataset()
