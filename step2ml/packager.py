"""
Stage 05: Package - Apache Parquet output writer and PyTorch Dataset DataLoader.
"""

from pathlib import Path
from typing import Dict, Any, List
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ParquetPackager:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def package_to_parquet(self, records: List[Dict[str, Any]], shard_id: int) -> Path:
        """Serializes validated CAD representations into Arrow/Parquet shards."""
        schema = pa.schema([
            ("part_id", pa.string()),
            ("source_file", pa.string()),
            ("dataset_source", pa.string()),
            ("file_hash", pa.string()),
            ("num_points", pa.int32()),
            ("points", pa.list_(pa.float32())),
            ("normals", pa.list_(pa.float32())),
            ("point_face_ids", pa.list_(pa.int32())),
            ("surface_area", pa.float32()),
            ("volume", pa.float32())
        ])

        data_dict = {
            "part_id": [],
            "source_file": [],
            "dataset_source": [],
            "file_hash": [],
            "num_points": [],
            "points": [],
            "normals": [],
            "point_face_ids": [],
            "surface_area": [],
            "volume": []
        }

        for r in records:
            meta = r["metadata"]
            geom = r["geometry"]

            data_dict["part_id"].append(meta.part_id)
            data_dict["source_file"].append(meta.source_file)
            data_dict["dataset_source"].append(meta.dataset_source)
            data_dict["file_hash"].append(meta.file_hash)
            data_dict["num_points"].append(len(geom["point_cloud"]))
            data_dict["points"].append(geom["point_cloud"].flatten().tolist())
            data_dict["normals"].append(geom["normals"].flatten().tolist())
            data_dict["point_face_ids"].append(geom["point_face_ids"].tolist())
            data_dict["surface_area"].append(geom["surface_area"])
            data_dict["volume"].append(geom["volume"])

        table = pa.Table.from_pydict(data_dict, schema=schema)
        out_file = self.output_dir / f"cad_dataset_shard_{shard_id:04d}.parquet"
        pq.write_table(table, out_file)
        return out_file


if TORCH_AVAILABLE:
    class StepMLPyTorchDataset(Dataset):
        """Native PyTorch Dataset wrapper that loads output Parquet shards directly for model training."""

        def __init__(self, parquet_paths: List[Path]):
            self.records = []
            for path in parquet_paths:
                table = pq.read_table(path)
                df = table.to_pandas()
                for _, row in df.iterrows():
                    num_pts = row["num_points"]
                    points = np.array(row["points"], dtype=np.float32).reshape(num_pts, 3)
                    normals = np.array(row["normals"], dtype=np.float32).reshape(num_pts, 3)
                    face_ids = np.array(row["point_face_ids"], dtype=np.int32)
                    self.records.append({
                        "part_id": row["part_id"],
                        "points": points,
                        "normals": normals,
                        "face_ids": face_ids
                    })

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            item = self.records[idx]
            return {
                "part_id": item["part_id"],
                "points": torch.from_numpy(item["points"]),
                "normals": torch.from_numpy(item["normals"]),
                "face_ids": torch.from_numpy(item["face_ids"])
            }