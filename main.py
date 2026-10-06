"""
STEP→ML Command Line Interface executable entrypoint.
"""

from pathlib import Path
import sys

from step2ml.config import PipelineConfig
from step2ml.pipeline import CADPipelineRunner
from step2ml.packager import StepMLPyTorchDataset
from examples.generate_sample_cad import create_sample_cad_files


def main():
    print("=" * 60)
    print(" STEP→ML — CAD-to-Training-Data Pipeline Execution")
    print("=" * 60)

    # 1. Prepare Paths & Test Data
    raw_dir = Path("./data/raw")
    output_dir = Path("./data/parquet_shards")

    print("[1/4] Preparing sample CAD files...")
    sample_files = create_sample_cad_files(raw_dir, count=4)
    print(f"      Created {len(sample_files)} source CAD files.")

    # 2. Configure Pipeline
    config = PipelineConfig(
        source_dir=raw_dir,
        output_dir=output_dir,
        shard_size=2,
        num_uniform_points=1024,
        num_feature_points=512,
        require_watertight=False
    )

    # 3. Execute Pipeline
    print("[2/4] Initializing pipeline runner...")
    runner = CADPipelineRunner(config)

    print("[3/4] Running batch processing...")
    results = runner.run_batch(sample_files)

    print("\n--- Pipeline Execution Summary ---")
    print(f"  Processed Files : {results['total_processed']}")
    print(f"  Shards Generated: {len(results['shards'])}")
    print(f"  Failed Files    : {len(results['failures'])}")

    # 4. Load Output using PyTorch Dataset
    if results['shards']:
        print("\n[4/4] Testing PyTorch Dataset Loader integration...")
        dataset = StepMLPyTorchDataset(results['shards'])
        print(f"      Dataset successfully loaded {len(dataset)} parts.")
        
        item = dataset[0]
        print("\nPyTorch Batch Spec Example:")
        print(f"  Part ID            : {item['part_id']}")
        print(f"  Point Cloud Tensor : {item['points'].shape} ({item['points'].dtype})")
        print(f"  Normals Tensor     : {item['normals'].shape} ({item['normals'].dtype})")
        print(f"  Face IDs Tensor    : {item['face_ids'].shape} ({item['face_ids'].dtype})")
        print("=" * 60)
        print("Pipeline run completed successfully.")


if __name__ == "__main__":
    main()