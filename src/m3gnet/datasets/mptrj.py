"""Materials Project Trajectory Dataset."""

import json
import math
import os
import pickle
from collections import defaultdict
from collections.abc import Callable, Iterator
from typing import Any

import ijson
import lmdb
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from loguru import logger
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from torch_geometric.data import Dataset, download_url
from tqdm import tqdm

from m3gnet.graph.converter import GraphConverter


class StatisticsTracker:
    """Track statistics of labels during processing."""

    def __init__(self):
        """Initializing."""
        self.stats = defaultdict(
            lambda: {
                "count": 0,
                "mean": 0.0,
                "M2": 0.0,
                "min": float("inf"),
                "max": float("-inf"),
                "values": [],  # For percentiles via reservoir sampling
            }
        )

    def update(self, labels: dict[str, Any], atoms: Atoms):
        """Update statistics with a new sample."""
        # Track scalar values
        for key in ["energy", "corrected_energy", "bandgap"]:
            if key in labels and labels[key] is not None:
                self._update_scalar_stat(key, float(labels[key]))

        # Track force magnitudes
        if "forces" in labels and labels["forces"] is not None:
            forces = np.array(labels["forces"])
            if forces.size > 0:
                force_magnitudes = np.linalg.norm(forces, axis=1)
                self._update_scalar_stat(
                    "force_magnitude", float(np.mean(force_magnitudes))
                )
                self._update_scalar_stat(
                    "max_force_magnitude", float(np.max(force_magnitudes))
                )

        # Track stress magnitudes
        if "stress" in labels and labels["stress"] is not None:
            stress = np.array(labels["stress"])
            if stress.shape == (3, 3):
                stress_magnitude = np.linalg.norm(stress, "fro")
                self._update_scalar_stat("stress_magnitude", float(stress_magnitude))

        # Track number of atoms
        self._update_scalar_stat("num_atoms", len(atoms))

        # Track atomic numbers distribution (we only care about the count)
        for atomic_num in atoms.get_atomic_numbers():
            stat = self.stats[f"element_{atomic_num}"]
            stat["count"] += 1

    def _update_scalar_stat(self, key: str, value: float):
        """Update statistics for a scalar value using Welford's algorithm."""
        stat = self.stats[key]
        stat["count"] += 1
        delta = value - stat["mean"]
        stat["mean"] += delta / stat["count"]
        delta2 = value - stat["mean"]
        stat["M2"] += delta * delta2
        stat["min"] = min(stat["min"], value)
        stat["max"] = max(stat["max"], value)

        # Reservoir sampling for percentiles (sample size: 10,000)
        if len(stat["values"]) < 10000:
            stat["values"].append(value)
        elif np.random.random() < 10000 / stat["count"]:
            idx_to_replace = np.random.randint(10000)
            stat["values"][idx_to_replace] = value

    def get_statistics(self) -> dict[str, Any]:
        """Finalize and return all computed statistics."""
        result = {}
        for key, stat in self.stats.items():
            if stat["count"] == 0:
                continue

            # For element counts, we only store the count
            if key.startswith("element_"):
                result[key] = {"count": stat["count"]}
                continue

            mean = stat["mean"]
            if stat["count"] > 1:
                variance = stat["M2"] / (stat["count"] - 1)
                std = math.sqrt(variance) if variance > 0 else 0.0
            else:
                std = 0.0

            result[key] = {
                "min": stat["min"],
                "max": stat["max"],
                "mean": mean,
                "std": std,
                "count": stat["count"],
            }

            # Add percentiles from our sample
            if len(stat["values"]) > 10:
                values_arr = np.array(stat["values"])
                result[key]["percentiles"] = {
                    p: float(np.percentile(values_arr, p)) for p in [25, 50, 75, 95, 99]
                }
        return result

    def load_from_existing(self, existing_stats: dict[str, Any]):
        """Load existing statistics to continue tracking."""
        for key, data in existing_stats.items():
            if not isinstance(data, dict) or "count" not in data:
                continue

            self.stats[key]["count"] = data.get("count", 0)
            if key.startswith("element_"):
                continue

            self.stats[key]["min"] = data.get("min", float("inf"))
            self.stats[key]["max"] = data.get("max", float("-inf"))
            self.stats[key]["mean"] = data.get("mean", 0.0)
            std = data.get("std", 0.0)
            count = data.get("count", 0)
            if count > 1:
                self.stats[key]["M2"] = (std**2) * (count - 1)


class MPTrjDataset(Dataset):
    """Materials Project Trajectory Dataset.

    Args:
        root (str): The root directory of the dataset.
        transform (Callable | None): A function/transform that takes in an
            ase.Atoms object and returns a transformed version.
        pre_transform (Callable | None): A function/transform that takes in
            an ase.Atoms object and returns a transformed version.
        pre_filter (Callable | None): A function/transform that takes in
            an ase.Atoms object and returns a boolean value.
        graph_converter (GraphConverter | None): A graph converter to convert
            ase.Atoms objects to PyTorch Geometric graphs.
        total_samples (int): The total number of samples in the dataset.
    """

    url = "https://figshare.com/ndownloader/files/41619375"

    def __init__(
        self,
        root: str = os.path.join(
            os.path.expanduser("~"), ".cache", "datasets", "mptrj"
        ),
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
        graph_converter: GraphConverter | None = None,
        total_samples: int = 1580395,
    ):
        """Initialize the Materials Project Trajectory Dataset."""
        self.graph_converter = graph_converter or GraphConverter()
        self._expected_total_samples = total_samples
        self._total_samples = 0
        self._db_envs = None  # For lazy loading of LMDB environments
        super().__init__(root, transform, pre_transform, pre_filter)
        self._total_samples = self.len()  # Load length after init

    @property
    def raw_file_names(self):
        """Get the names of the raw files."""
        return ["MPtrj_2022.9_full.json"]

    @property
    def processed_file_names(self):
        """Get the names of the processed files."""
        status = self.get_processing_status()
        if status.get("is_processed", False):
            # If complete, all shards are "processed files"
            return [f"shard_{i:04d}" for i in range(status.get("num_shards", 0))]
        # If not complete, PyG thinks processing is not done
        return []

    def download(self):
        """Download the raw data file."""
        download_url(self.url, self.raw_dir, filename=self.raw_file_names[0])

    def get_processing_status(self) -> dict[str, Any]:
        """Get the current processing status."""
        status = {
            "is_processed": False,
            "samples_processed": 0,
            "total_expected": self._expected_total_samples,
            "completion_percentage": 0.0,
            "num_shards": 0,
            "separate_labels": None,
            "has_statistics": False,
            "last_processed_mp_id": None,
            "last_processed_frame_id": None,
        }
        first_shard_path = os.path.join(self.processed_dir, "shard_0000", "graphs")
        if not os.path.exists(first_shard_path):
            return status

        try:
            with (
                lmdb.open(first_shard_path, readonly=True) as graph_db,
                graph_db.begin() as txn,
            ):
                metadata_bytes = txn.get(b"__metadata__")
                if metadata_bytes:
                    metadata = pickle.loads(metadata_bytes)  # noqa: S301
                    status.update(
                        {
                            "samples_processed": metadata.get("total_samples", 0),
                            "num_shards": metadata.get("num_shards", 0),
                            "separate_labels": metadata.get("separate_labels"),
                            "last_processed_mp_id": metadata.get(
                                "last_processed_mp_id"
                            ),
                            "last_processed_frame_id": metadata.get(
                                "last_processed_frame_id"
                            ),
                            "is_processed": metadata.get("processing_complete", False),
                        }
                    )
                    if self._expected_total_samples > 0:
                        status["completion_percentage"] = (
                            status["samples_processed"] / self._expected_total_samples
                        ) * 100

            stats_path = os.path.join(self.processed_dir, "statistics")
            status["has_statistics"] = os.path.exists(stats_path)
        except Exception as e:
            logger.warning(f"Error reading processing status: {e}")
        return status

    def print_status(self):
        """Prints the current processing status of the dataset."""
        status = self.get_processing_status()
        logger.info("--- Dataset Status ---")
        logger.info(f"Root directory: {self.root}")
        logger.info(f"Processed: {status['is_processed']}")
        logger.info(
            f"Samples: {status['samples_processed']:,} / {status['total_expected']:,} "
            f"({status['completion_percentage']:.2f}%)"
        )
        logger.info(f"Number of shards: {status['num_shards']}")
        logger.info(f"Statistics available: {status['has_statistics']}")
        if status["last_processed_mp_id"]:
            logger.info(
                "Last sample processed: "
                f"{status['last_processed_mp_id']}/{status['last_processed_frame_id']}"
            )
        logger.info("----------------------")

    def _find_resume_point(self) -> tuple[str | None, str | None, int]:
        """Find where to resume processing from."""
        status = self.get_processing_status()
        return (
            status["last_processed_mp_id"],
            status["last_processed_frame_id"],
            status["samples_processed"],
        )

    def _should_skip_sample(
        self,
        mp_id: str,
        frame_id: str,
        last_mp_id: str | None,
        last_frame_id: str | None,
    ) -> bool:
        """Determine if a sample should be skipped during resume."""
        if last_mp_id is None or last_frame_id is None:
            return False
        # Assumes lexicographical comparison works for mp_id and frame_id
        return mp_id < last_mp_id or mp_id == last_mp_id and frame_id <= last_frame_id

    def _stream_json_entries(
        self, json_file_path: str
    ) -> Iterator[tuple[str, str, dict[str, Any]]]:
        """Stream JSON entries using ijson for robustness and efficiency."""
        logger.info(f"Streaming JSON data from {json_file_path} using ijson...")
        try:
            with open(json_file_path, "rb") as f:
                # The path is mp_id -> frame_id -> value
                parser = ijson.kvitems(f, "", use_float=True)
                for mp_id, mat_data in parser:
                    for frame_id, values in mat_data.items():
                        yield mp_id, frame_id, values
        except Exception as e:
            logger.error(
                f"Failed to stream with ijson: {e}. "
                "Falling back to full load (might use a lot of memory)."
            )
            with open(json_file_path) as f:
                data = json.load(f)
            for mp_id, mat_data in data.items():
                for frame_id, values in mat_data.items():
                    yield mp_id, frame_id, values

    def _setup_processing_environment(
        self, force_reprocess, separate_labels, max_shard_size_gb, compute_statistics
    ):
        """Handles the initial checks and setup for processing."""
        status = self.get_processing_status()

        if not force_reprocess and status["is_processed"]:
            logger.info("✅ Dataset already fully processed.")
            return None  # Signal to stop processing

        if not force_reprocess and status["samples_processed"] > 0:
            logger.info(
                f"🔄 Resuming processing from {status['samples_processed']:,} samples "
                f"({status['completion_percentage']:.1f}% complete)."
            )
            last_mp_id, last_frame_id, samples_already_processed = (
                self._find_resume_point()
            )
        else:
            logger.info("🆕 Starting fresh processing...")
            samples_already_processed, last_mp_id, last_frame_id = 0, None, None

        max_shard_size = int(max_shard_size_gb * 1024 * 1024 * 1024)
        self.graph_shards = []
        self.label_shards = [] if separate_labels else None
        self.current_shard_idx = 0
        self._create_new_shard(separate_labels, max_shard_size)

        if compute_statistics:
            self.stats_tracker = StatisticsTracker()
            # To implement full resume, you would load existing stats here.
            # self._load_existing_statistics()

        return samples_already_processed, last_mp_id, last_frame_id, max_shard_size

    def _finalize_processing(
        self,
        sample_count,
        separate_labels,
        compute_statistics,
        last_mp_id,
        last_frame_id,
    ):
        """Handles saving final metadata, stats, and logging."""
        self._save_metadata(
            sample_count, separate_labels, True, last_mp_id, last_frame_id
        )
        if compute_statistics:
            self._save_statistics()
            logger.info("📊 Statistics computed and saved.")

        self._total_samples = sample_count
        logger.info(f"✅ Processing complete! Total samples: {self._total_samples:,}")

    def process(
        self,
        chunk_size: int = 10_000,
        max_samples: int | None = None,
        separate_labels: bool = True,
        max_shard_size_gb: float = 10.0,
        compute_statistics: bool = True,
        force_reprocess: bool = False,
    ):
        """Process raw data, converting it into sharded LMDB databases."""
        setup_result = self._setup_processing_environment(
            force_reprocess, separate_labels, max_shard_size_gb, compute_statistics
        )
        if setup_result is None:
            return  # Processing is already complete

        samples_already_processed, last_mp_id, last_frame_id, max_shard_size = (
            setup_result
        )

        try:
            sample_count = samples_already_processed
            chunk_data, chunk_labels = [], []
            latest_mp_id, latest_frame_id = last_mp_id, last_frame_id

            json_path = os.path.join(self.raw_dir, self.raw_file_names[0])
            total_for_pbar = (
                max_samples if max_samples is not None else self._expected_total_samples
            )

            pbar = tqdm(
                self._stream_json_entries(json_path),
                desc="Processing structures",
                unit=" samples",
                total=total_for_pbar,
                initial=samples_already_processed,
            )

            for mp_id, frame_id, values in pbar:
                if self._should_skip_sample(mp_id, frame_id, last_mp_id, last_frame_id):
                    continue

                if max_samples and sample_count >= max_samples:
                    break

                try:
                    structure = Structure.from_dict(values["structure"])
                    atoms = AseAtomsAdaptor.get_atoms(structure)

                    energy, forces, stress = (
                        values["uncorrected_total_energy"],
                        values["force"],
                        values["stress"],
                    )
                    atoms.calc = SinglePointCalculator(
                        atoms=atoms, energy=energy, forces=forces, stress=stress
                    )

                    graph_data = self.graph_converter.convert_ase_atoms(atoms)
                    labels = {
                        k: values.get(k) for k in ["corrected_total_energy", "bandgap"]
                    }
                    labels.update(
                        {
                            "energy": energy,
                            "forces": forces,
                            "stress": stress,
                            "mp_id": mp_id,
                            "frame_id": frame_id,
                        }
                    )

                    if compute_statistics:
                        self.stats_tracker.update(labels, atoms)

                    chunk_data.append(graph_data)
                    chunk_labels.append(labels)
                    sample_count += 1
                    latest_mp_id, latest_frame_id = mp_id, frame_id

                    if len(chunk_data) >= chunk_size:
                        self._save_chunk_to_shards(
                            chunk_data,
                            chunk_labels,
                            sample_count - len(chunk_data),
                            separate_labels,
                            max_shard_size,
                        )
                        self._save_metadata(
                            sample_count,
                            separate_labels,
                            False,
                            latest_mp_id,
                            latest_frame_id,
                        )
                        chunk_data.clear()
                        chunk_labels.clear()

                except Exception as e:
                    logger.warning(f"Failed to process {mp_id}/{frame_id}: {e}")
                    continue

            pbar.close()

            # Save any remaining data
            if chunk_data:
                self._save_chunk_to_shards(
                    chunk_data,
                    chunk_labels,
                    sample_count - len(chunk_data),
                    separate_labels,
                    max_shard_size,
                )

            # Finalize
            self._finalize_processing(
                sample_count,
                separate_labels,
                compute_statistics,
                latest_mp_id,
                latest_frame_id,
            )

        finally:
            self._close_all_shards()

    def _create_new_shard(self, separate_labels: bool, max_shard_size: int):
        """Create a new LMDB shard."""
        shard_dir = os.path.join(
            self.processed_dir, f"shard_{self.current_shard_idx:04d}"
        )
        os.makedirs(shard_dir, exist_ok=True)

        graph_shard_path = os.path.join(shard_dir, "graphs")
        graph_shard = lmdb.open(graph_shard_path, map_size=max_shard_size)
        self.graph_shards.append(graph_shard)

        if separate_labels:
            label_shard_path = os.path.join(shard_dir, "labels")
            label_shard = lmdb.open(
                label_shard_path, map_size=max_shard_size // 5
            )  # Labels are smaller
            self.label_shards.append(label_shard)

        self.samples_in_current_shard = 0
        logger.info(f"Created shard {self.current_shard_idx} at {shard_dir}")

    def _save_chunk_to_shards(
        self,
        chunk_data: list,
        chunk_labels: list,
        start_idx: int,
        separate_labels: bool,
        max_shard_size: int,
    ):
        """Save a chunk of data, handling shard size limits gracefully."""
        samples_in_chunk = len(chunk_data)
        saved_count = 0

        while saved_count < samples_in_chunk:
            try:
                current_graph_shard = self.graph_shards[self.current_shard_idx]
                current_label_shard = (
                    self.label_shards[self.current_shard_idx]
                    if separate_labels
                    else None
                )

                index_mappings = []

                # --- Transaction for Data ---
                with current_graph_shard.begin(write=True) as graph_txn:
                    label_txn = (
                        current_label_shard.begin(write=True)
                        if separate_labels and current_label_shard
                        else None
                    )
                    try:
                        for i in range(saved_count, samples_in_chunk):
                            global_idx = start_idx + i
                            local_idx = self.samples_in_current_shard + (
                                i - saved_count
                            )
                            key = str(local_idx).encode()

                            graph_data_to_save, labels_to_save = (
                                chunk_data[i],
                                chunk_labels[i],
                            )

                            if separate_labels:
                                graph_txn.put(key, pickle.dumps(graph_data_to_save))
                                if label_txn:
                                    label_txn.put(key, pickle.dumps(labels_to_save))
                            else:
                                combined = {
                                    "graph": graph_data_to_save,
                                    "labels": labels_to_save,
                                }
                                graph_txn.put(key, pickle.dumps(combined))

                            index_key = f"global_{global_idx}".encode()
                            mapping = f"{self.current_shard_idx}:{local_idx}".encode()
                            index_mappings.append((index_key, mapping))

                        if label_txn:
                            label_txn.commit()

                    except Exception:
                        if label_txn:
                            label_txn.abort()
                        raise

                # --- Separate Transaction for Index ---
                with self.graph_shards[0].begin(write=True) as index_txn:
                    for key, value in index_mappings:
                        index_txn.put(key, value)

                num_written = samples_in_chunk - saved_count
                self.samples_in_current_shard += num_written
                saved_count += num_written

            except lmdb.MapFullError:
                logger.warning(
                    f"Shard {self.current_shard_idx} is full with "
                    f"{self.samples_in_current_shard} samples. "
                    "Creating a new one."
                )
                self.current_shard_idx += 1
                self._create_new_shard(separate_labels, max_shard_size)
            except Exception as e:
                logger.error(f"An unexpected error occurred during saving: {e}")
                raise

    def _save_metadata(
        self,
        total_samples: int,
        separate_labels: bool,
        is_complete: bool,
        last_mp_id: str | None,
        last_frame_id: str | None,
    ):
        """Save dataset metadata to the first shard."""
        metadata = {
            "total_samples": total_samples,
            "separate_labels": separate_labels,
            "num_shards": len(self.graph_shards),
            "last_processed_mp_id": last_mp_id,
            "last_processed_frame_id": last_frame_id,
            "processing_complete": is_complete,
            "version": "2.2",  # Incremented version for the fix
        }
        if self.graph_shards:
            with self.graph_shards[0].begin(write=True) as txn:
                txn.put(b"__metadata__", pickle.dumps(metadata))

    def _save_statistics(self):
        """Save computed statistics to a separate LMDB database."""
        stats = self.stats_tracker.get_statistics()
        stats_db_path = os.path.join(self.processed_dir, "statistics")
        os.makedirs(os.path.dirname(stats_db_path), exist_ok=True)
        with (
            lmdb.open(stats_db_path, map_size=1 * 1024 * 1024 * 1024) as stats_db,
            stats_db.begin(write=True) as txn,
        ):
            txn.put(b"statistics", pickle.dumps(stats))

    def _close_all_shards(self):
        """Close all LMDB shards used during processing."""
        for shard in self.graph_shards:
            shard.close()
        if self.label_shards:
            for shard in self.label_shards:
                shard.close()

    def len(self):
        """Return the number of graphs in the dataset."""
        if self._total_samples:
            return self._total_samples

        status = self.get_processing_status()
        self._total_samples = status.get("samples_processed", 0)
        return self._total_samples

    def _find_shard_for_sample(self, idx: int) -> tuple[int, int]:
        """Find which shard contains the given sample index."""
        first_shard_path = os.path.join(self.processed_dir, "shard_0000", "graphs")
        try:
            with (
                lmdb.open(first_shard_path, readonly=True, lock=False) as db,
                db.begin() as txn,
            ):
                key = f"global_{idx}".encode()
                mapping_bytes = txn.get(key)
                if mapping_bytes:
                    shard_idx_str, local_idx_str = mapping_bytes.decode().split(":")
                    return int(shard_idx_str), int(local_idx_str)
        except Exception as e:
            logger.error(f"An error occurred while finding shard for sample {idx}: {e}")

        raise IndexError(
            f"Sample {idx} not found in global index. "
            "The index may be incomplete or corrupt."
        )

    def _open_dbs(self):
        """Open all LMDB environments and keep them for fast access."""
        if self._db_envs is not None:
            return

        logger.info("Lazy-loading LMDB environments for reading...")
        self._db_envs = {"graphs": [], "labels": []}
        status = self.get_processing_status()
        num_shards = status.get("num_shards", 0)
        separate_labels = status.get("separate_labels", True)

        for shard_idx in range(num_shards):
            shard_dir = os.path.join(self.processed_dir, f"shard_{shard_idx:04d}")
            graph_path = os.path.join(shard_dir, "graphs")
            if os.path.exists(graph_path):
                graph_db = lmdb.open(
                    graph_path,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False,
                )
                self._db_envs["graphs"].append(graph_db)

            if separate_labels:
                label_path = os.path.join(shard_dir, "labels")
                if os.path.exists(label_path):
                    label_db = lmdb.open(
                        label_path,
                        readonly=True,
                        lock=False,
                        readahead=False,
                        meminit=False,
                    )
                    self._db_envs["labels"].append(label_db)

        logger.info(
            f"Opened {len(self._db_envs['graphs'])} graph shards "
            f"and {len(self._db_envs['labels'])} label shards."
        )

    def close(self):
        """Close all opened LMDB environments used for reading."""
        if self._db_envs is not None:
            for db_list in self._db_envs.values():
                for db in db_list:
                    db.close()
            self._db_envs = None
            logger.info("Closed all LMDB environments.")

    def get(self, idx: int) -> dict[str, Any]:
        """Get the idx-th graph in the dataset efficiently."""
        self._open_dbs()

        if self._db_envs is None or not self._db_envs["graphs"]:
            raise RuntimeError("Dataset not processed or shards not found.")

        try:
            shard_idx, local_idx = self._find_shard_for_sample(idx)
            key = str(local_idx).encode()

            with self._db_envs["graphs"][shard_idx].begin() as txn:
                data_bytes = txn.get(key)
            if data_bytes is None:
                raise IndexError(
                    f"Sample {idx} (local {local_idx}) "
                    f"not found in graph shard {shard_idx}"
                )
            data = pickle.loads(data_bytes)  # noqa: S301

            separate_labels = bool(self._db_envs["labels"])
            if separate_labels:
                with self._db_envs["labels"][shard_idx].begin() as txn:
                    label_bytes = txn.get(key)
                if label_bytes is None:
                    raise IndexError(
                        f"Labels for sample {idx} not found in shard {shard_idx}"
                    )
                labels = pickle.loads(label_bytes)  # noqa: S301
                graph_data = data
            else:
                graph_data = data["graph"]
                labels = data["labels"]

            if self.transform:
                graph_data = self.transform(graph_data)

            return {"graph": graph_data, "labels": labels}

        except Exception as e:
            logger.error(f"Error loading sample {idx}: {e}")
            raise


if __name__ == "__main__":
    logger.info("Initializing MPTrjDataset...")
    # Create an instance of the dataset in a specific root directory
    dataset = MPTrjDataset()

    # Print the current status (processed, how many samples, etc.)
    dataset.print_status()

    # --- To Process the Dataset (run this once) ---
    logger.info("Starting dataset processing...")
    try:
        dataset.process(
            chunk_size=5000,
            max_shard_size_gb=5.0,
            compute_statistics=True,
            force_reprocess=False,
        )
        logger.info("Processing finished successfully.")
    except Exception as main_exc:
        logger.error(f"An error occurred during processing: {main_exc}")

    # --- To Use the Processed Dataset ---
    status = dataset.get_processing_status()
    if status["is_processed"]:
        logger.info("\n--- Using the processed dataset ---")
        try:
            print(f"Total dataset length: {len(dataset)}")
            print("Fetching sample #0...")
            data_point = dataset[0]
            print("Sample #0 keys:", data_point.keys())
            print("Graph keys:", data_point["graph"].keys())
            print("Labels:", data_point["labels"])
            dataset.close()
            logger.info("Dataset handles closed.")
        except Exception as use_exc:
            logger.error(f"An error occurred while using the dataset: {use_exc}")
    else:
        logger.warning(
            "\nDataset is not fully processed. "
            "Run the processing step to use the dataset."
        )

    # --- To Query the Dataset ---
    logger.info("Querying the dataset...")
    try:
        query_result = dataset.get(100001)
        logger.info(f"Query result: {query_result}")
        logger.info(f"Query result keys: {query_result.keys()}")
        logger.info(f"Graph keys: {query_result['graph'].keys()}")
        logger.info(f"Labels keys: {query_result['labels'].keys()}")
        logger.info(f"Energy: {query_result['labels']['energy']}")
        logger.info(f"Forces: {query_result['labels']['forces']}")
        logger.info(f"Stress: {query_result['labels']['stress']}")
        logger.info(f"MP ID: {query_result['labels']['mp_id']}")
        logger.info(f"Frame ID: {query_result['labels']['frame_id']}")
    except Exception as query_exc:
        logger.error(f"An error occurred while querying the dataset: {query_exc}")
