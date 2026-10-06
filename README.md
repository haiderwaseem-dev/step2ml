
## 💡 Overview

Parametric B-Rep files (STEP/IGES) are built for CAD software, not neural networks. Before 3D Machine Learning models can learn geometry, thousands of inconsistent CAD files must be reliably converted into uniform point clouds, normal vectors, and metadata without breaking batch training execution.

`STEP→ML` solves this with an isolated, 5-stage pipeline featuring fallback B-Rep/mesh engines, curvature-weighted point sampling, geometric validation, deduplication, and high-performance Parquet sharding.

---

## 🏗️ Pipeline Architecture

┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  SOURCE  │ ──► │  PARSE   │ ──► │ CONVERT  │ ──► │ VALIDATE │ ──► │ PACKAGE  │
└──────────┘     └──────────┘     └──────────┘     └──────────┘     └──────────┘
Provenance        OpenCASCADE /    Mesh + Point     Watertight,       Parquet +
& SHA256          Topology        Cloud Sampling   Dedup Hash        PyTorch


1. **Source**: Ingests CAD files, generates SHA256 hashes for data lineage, and attaches dataset provenance metadata.
2. **Parse**: Uses `pythonOCC` (OpenCASCADE B-Rep topology) with a fallback mesh parser to extract topological faces, vertices, and boundary representations.
3. **Convert**: Tessellates surfaces into indexed triangle meshes and samples uniform + curvature-aware feature point clouds while preserving face associations.
4. **Validate**: Filters out non-manifold geometry, verifies volume thresholds, and deduplicates identical parts via spatial-histogram geometric hashing.
5. **Package**: Exports versioned, columnar Apache Parquet shards ready for immediate consumption by a native `PyTorch` Dataset class.

---

## 📁 Directory Structure

```text
step2ml/
├── data/                  # Local directory for raw CAD inputs and Parquet shards
├── examples/              # Scripts to generate synthetic 3D CAD sample geometries
├── step2ml/               # Core pipeline package
│   ├── config.py          # Pydantic configuration schemas
│   ├── source.py          # Stage 01: Ingestion & Provenance
│   ├── parser.py          # Stage 02: B-Rep & Mesh Parsing
│   ├── converter.py       # Stage 03: Point Cloud & Curvature Sampling
│   ├── validator.py       # Stage 04: Geometric Validation & Hash Dedup
│   ├── packager.py        # Stage 05: Parquet Writer & PyTorch Dataset
│   └── pipeline.py       # Batch Runner Orchestrator
├── tests/                 # Automated pytest suite
├── app.py                 # Streamlit 3D interactive web visualizer
├── main.py                # Pipeline CLI entrypoint
├── requirements.txt       # Project dependencies
└── setup.py               # Package setup script
🚀 Quick Start
1. Clone & Setup Environment
Bash
git clone [https://github.com/haiderwaseem-dev/step2ml.git](https://github.com/haiderwaseem-dev/step2ml.git)
cd step2ml

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
2. Run Pipeline Execution
Bash
python main.py
3. Run Automated Tests
Bash
python -m pytest tests/ -v
🖥️ Streamlit 3D Interactive Visualizer
Launch the interactive web application locally to upload 3D models and visually inspect point cloud extractions and normal vectors:

Bash
streamlit run app.py
⚡ PyTorch DataLoader Integration
step2ml includes a native PyTorch Dataset loader (StepMLPyTorchDataset) that loads output Parquet shards directly into PyTorch tensors without intermediate conversion steps:

Python
from step2ml.packager import StepMLPyTorchDataset
from torch.utils.data import DataLoader

# Load generated Parquet shards
dataset = StepMLPyTorchDataset(["./data/parquet_shards/cad_dataset_shard_0000.parquet"])
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

for batch in dataloader:
    points = batch["points"]    # Shape: [batch_size, num_points, 3]
    normals = batch["normals"]  # Shape: [batch_size, num_points, 3]
    print(f"Loaded batch of shape: {points.shape}")
