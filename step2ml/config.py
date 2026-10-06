"""
Pipeline Configuration module.
"""

from pathlib import Path
from pydantic import BaseModel, Field


class PipelineConfig(BaseModel):
    # Paths
    source_dir: Path = Field(default=Path("./data/raw"))
    processed_dir: Path = Field(default=Path("./data/processed"))
    output_dir: Path = Field(default=Path("./data/parquet_shards"))

    # Conversion settings
    num_uniform_points: int = Field(default=2048, description="Number of uniform surface points to sample")
    num_feature_points: int = Field(default=1024, description="Number of high-curvature/feature points to sample")
    tessellation_tolerance: float = Field(default=0.1, description="Mesh deflection tolerance for OCC tessellation")

    # Validation thresholds
    require_watertight: bool = Field(default=True)
    min_volume: float = Field(default=1e-6)
    max_bounding_box_diag: float = Field(default=1000.0)

    # Packaging settings
    shard_size: int = Field(default=500, description="Number of items per Parquet file shard")

    class Config:
        arbitrary_types_allowed = True