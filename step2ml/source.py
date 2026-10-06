"""
Stage 01: Source - Ingestion, provenance tracking, and metadata extraction.
"""

import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional


@dataclass
class PartMetadata:
    part_id: str
    source_file: str
    dataset_source: str
    license_type: str
    file_hash: str
    file_size_bytes: int


class SourceIngestor:
    def __init__(self, dataset_source: str = "ABC_Dataset", default_license: str = "CC-BY-4.0"):
        self.dataset_source = dataset_source
        self.default_license = default_license

    def compute_sha256(self, filepath: Path) -> str:
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()

    def process_file(self, filepath: Path, custom_metadata: Optional[Dict[str, Any]] = None) -> PartMetadata:
        if not filepath.exists():
            raise FileNotFoundError(f"CAD source file not found: {filepath}")

        file_hash = self.compute_sha256(filepath)
        part_id = f"part_{file_hash[:12]}"

        metadata = PartMetadata(
            part_id=part_id,
            source_file=filepath.name,
            dataset_source=self.dataset_source,
            license_type=self.default_license,
            file_hash=file_hash,
            file_size_bytes=filepath.stat().st_size
        )

        return metadata