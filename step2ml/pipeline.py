"""
Pipeline Orchestration engine with task logging and error resilience.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any

from step2ml.config import PipelineConfig
from step2ml.source import SourceIngestor
from step2ml.parser import CADParser
from step2ml.converter import MeshToPointCloudConverter
from step2ml.validator import GeometryValidator
from step2ml.packager import ParquetPackager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("STEP2ML")


class CADPipelineRunner:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.source_stage = SourceIngestor()
        self.parser_stage = CADParser(deflection_tolerance=config.tessellation_tolerance)
        self.converter_stage = MeshToPointCloudConverter(
            num_uniform=config.num_uniform_points,
            num_feature=config.num_feature_points
        )
        self.validator_stage = GeometryValidator(
            require_watertight=config.require_watertight,
            min_volume=config.min_volume,
            max_bounding_box_diag=config.max_bounding_box_diag
        )
        self.packager_stage = ParquetPackager(output_dir=config.output_dir)

    def run_batch(self, file_paths: List[Path]) -> Dict[str, Any]:
        valid_records = []
        failure_log = []
        shard_counter = 0
        written_shards = []

        logger.info(f"Starting pipeline execution for {len(file_paths)} STEP files...")

        for filepath in file_paths:
            try:
                # Stage 01: Source
                metadata = self.source_stage.process_file(filepath)

                # Stage 02: Parse
                parsed = self.parser_stage.parse_step(filepath)

                # Stage 03: Convert
                converted = self.converter_stage.convert(parsed)

                # Stage 04: Validate
                is_valid, err_code, geo_hash = self.validator_stage.validate(converted)

                if not is_valid:
                    logger.warning(f"Validation REJECT [{filepath.name}]: {err_code}")
                    failure_log.append({"file": str(filepath), "error": err_code})
                    continue

                valid_records.append({
                    "metadata": metadata,
                    "geometry": converted
                })
                logger.info(f"Successfully processed & validated: {filepath.name}")

                # Stage 05: Package Shard Check
                if len(valid_records) >= self.config.shard_size:
                    shard_file = self.packager_stage.package_to_parquet(valid_records, shard_counter)
                    written_shards.append(shard_file)
                    shard_counter += 1
                    valid_records.clear()

            except Exception as e:
                logger.error(f"Failed processing [{filepath.name}]: {str(e)}")
                failure_log.append({"file": str(filepath), "error": str(e)})

        # Flush remaining records
        if valid_records:
            shard_file = self.packager_stage.package_to_parquet(valid_records, shard_counter)
            written_shards.append(shard_file)

        logger.info(f"Pipeline complete. Created {len(written_shards)} Parquet shards. Rejections: {len(failure_log)}")

        return {
            "shards": written_shards,
            "failures": failure_log,
            "total_processed": len(file_paths)
        }