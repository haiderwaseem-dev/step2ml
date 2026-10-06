"""
Stage 02: Parse - OpenCASCADE / B-Rep Parser with geometric boundary fallback.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

try:
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopLoc import TopLoc_Location
    OCC_AVAILABLE = True
except ImportError:
    OCC_AVAILABLE = False

import trimesh


class CADParser:
    def __init__(self, deflection_tolerance: float = 0.1):
        self.deflection_tolerance = deflection_tolerance

    def parse_step(self, step_path: Path) -> Dict[str, Any]:
        """
        Extracts face topology and converts B-Rep to indexed triangular mesh primitives
        preserving surface-face associations.
        """
        if OCC_AVAILABLE:
            return self._parse_with_occ(step_path)
        else:
            return self._parse_with_fallback(step_path)

    def _parse_with_occ(self, step_path: Path) -> Dict[str, Any]:
        reader = STEPControl_Reader()
        status = reader.ReadFile(str(step_path))
        if status != 1:
            raise ValueError(f"Failed to read STEP file: {step_path}")

        reader.TransferRoots()
        shape = reader.OneShape()

        # Tessellate
        mesh_builder = BRepMesh_IncrementalMesh(shape, self.deflection_tolerance)
        mesh_builder.Perform()

        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        faces_data = []
        face_index = 0

        vertices_list = []
        triangles_list = []
        face_ids_list = []
        vert_offset = 0

        while explorer.More():
            occ_face = explorer.Current()
            loc = TopLoc_Location()
            triangulation = BRep_Tool.Triangulation(occ_face, loc)

            if triangulation is not None:
                num_nodes = triangulation.NbNodes()
                num_triangles = triangulation.NbTriangles()

                # Extract vertices
                nodes = np.zeros((num_nodes, 3))
                for i in range(1, num_nodes + 1):
                    p = triangulation.Node(i)
                    if not loc.IsIdentity():
                        p.Transform(loc.Transformation())
                    nodes[i - 1] = [p.X(), p.Y(), p.Z()]

                # Extract indices
                triangles = []
                for i in range(1, num_triangles + 1):
                    t = triangulation.Triangle(i)
                    n1, n2, n3 = t.Get()
                    triangles.append([n1 - 1 + vert_offset, n2 - 1 + vert_offset, n3 - 1 + vert_offset])

                vertices_list.append(nodes)
                if triangles:
                    triangles_list.append(np.array(triangles))
                    face_ids_list.extend([face_index] * len(triangles))

                vert_offset += num_nodes
                face_index += 1

            explorer.Next()

        all_vertices = np.vstack(vertices_list) if vertices_list else np.empty((0, 3))
        all_triangles = np.vstack(triangles_list) if triangles_list else np.empty((0, 3), dtype=int)

        mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_triangles, process=False)

        return {
            "mesh": mesh,
            "num_faces": face_index,
            "face_ids": np.array(face_ids_list)
        }

    def _parse_with_fallback(self, step_path: Path) -> Dict[str, Any]:
        """Fallback for environments without pythonocc installed."""
        loaded = trimesh.load(step_path, force='mesh')
        if isinstance(loaded, trimesh.Scene):
            mesh = loaded.dump(concatenate=True)
        else:
            mesh = loaded

        # Synthetic face clustering based on normal angles to preserve face metadata structure
        if len(mesh.faces) > 0:
            face_ids = trimesh.graph.connected_components(
                mesh.face_adjacency, min_len=1
            )
            mapped_face_ids = np.zeros(len(mesh.faces), dtype=int)
            for idx, comp in enumerate(face_ids):
                mapped_face_ids[comp] = idx
            num_faces = len(face_ids)
        else:
            mapped_face_ids = np.array([])
            num_faces = 0

        return {
            "mesh": mesh,
            "num_faces": num_faces,
            "face_ids": mapped_face_ids
        }