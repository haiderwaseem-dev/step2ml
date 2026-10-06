"""
Utility script generating synthetic 3D geometry CAD files for evaluation.
"""

from pathlib import Path
import trimesh


def create_sample_cad_files(output_dir: Path, count: int = 5) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    created_files = []

    for i in range(count):
        if i % 2 == 0:
            mesh = trimesh.creation.box(extents=(10, 20, 30))
        else:
            mesh = trimesh.creation.cylinder(radius=5, height=15)

        # Export as OBJ/STL (acting as test 3D CAD inputs)
        file_path = output_dir / f"sample_part_{i + 1:03d}.stl"
        mesh.export(file_path)
        created_files.append(file_path)

    return created_files


if __name__ == "__main__":
    paths = create_sample_cad_files(Path("./data/raw"), count=3)
    print(f"Generated {len(paths)} sample files in ./data/raw")