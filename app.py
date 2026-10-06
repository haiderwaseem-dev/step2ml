import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from step2ml.config import PipelineConfig
from step2ml.pipeline import CADPipelineRunner

st.set_page_config(page_title="STEP→ML Pipeline Visualizer", layout="wide")
st.title("STEP→ML — CAD-to-Training-Data Pipeline")

uploaded_file = st.sidebar.file_uploader("Upload 3D CAD File (STL/OBJ)", type=["stl", "obj"])

if uploaded_file:
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = Path(tmp_dir) / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        config = PipelineConfig(
            source_dir=Path(tmp_dir),
            output_dir=Path(tmp_dir) / "out",
            num_uniform_points=2048,
            num_feature_points=1024,
            require_watertight=False
        )

        runner = CADPipelineRunner(config)
        res = runner.run_batch([file_path])

        if res["shards"]:
            shard_path = res["shards"][0]
            df = pd.read_parquet(shard_path)
            pts = np.array(df["points"].iloc[0]).reshape(-1, 3)
            normals = np.array(df["normals"].iloc[0]).reshape(-1, 3)

            st.success(f"Parsed & Processed: {uploaded_file.name}")

            col1, col2 = st.columns([3, 1])

            with col1:
                st.subheader("Extracted Point Cloud Representation")
                fig = px.scatter_3d(
                    x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                    color=normals[:, 2],
                    opacity=0.7,
                    title="Sampled Surface Point Cloud (Color by Z-Normal)"
                )
                fig.update_traces(marker=dict(size=2))
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Extracted ML Features")
                st.json({
                    "Part ID": df["part_id"].iloc[0],
                    "File Hash": df["file_hash"].iloc[0][:12],
                    "Point Count": int(df["num_points"].iloc[0]),
                    "Surface Area": round(float(df["surface_area"].iloc[0]), 2),
                })
else:
    st.info("Upload a 3D CAD mesh file on the sidebar to preview the pipeline transformation.")