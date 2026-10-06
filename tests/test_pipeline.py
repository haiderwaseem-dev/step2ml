"""
End-to-End pytest execution test suite.
"""

from pathlib import Path
import pytest
import numpy as np

from step2ml.config import PipelineConfig
from step2ml.pipeline import CADPipelineRunner
from step2ml.packager import StepMLPyTorchDataset
from examples.generate_sample_cad import create_sample_cad_files


@pytest.fixture
def temp_environment(tmp_path):
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "shards"
    files = create_sample_cad_files(raw_dir, count=2)
    config = PipelineConfig(
        source_dir=raw_dir,
        output_dir=out_dir,
        shard_size=2,
        num_uniform_points=512,
        num_feature_points=256,
        require_watertight=False  # Disabled for synthetic test STL compatibility
    )
    return config, files


def test_full_pipeline_run(temp_environment):
    config, files = temp_environment
    runner = CADPipelineRunner(config)
    result = runner.run_batch(files)

    assert result["total_processed"] == 2
    assert len(result["shards"]) == 1
    assert Path(result["shards"][0]).exists()

    # Verify PyTorch Dataset Loader
    dataset = StepMLPyTorchDataset(result["shards"])
    assert len(dataset) == 2

    sample = dataset[0]
    assert "points" in sample
    assert "normals" in sample
    assert sample["points"].shape == (768, 3)  # 512 + 256
    assert sample["normals"].shape == (768, 3)