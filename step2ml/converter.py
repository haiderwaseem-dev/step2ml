"""
Stage 03: Convert - Tessellation, Point Cloud Sampling, Curvature Feature Sampling.
"""

from typing import Dict, Any, Tuple
import numpy as np
import trimesh


class MeshToPointCloudConverter:
    def __init__(self, num_uniform: int = 2048, num_feature: int = 1024):
        self.num_uniform = num_uniform
        self.num_feature = num_feature

    def convert(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        mesh: trimesh.Trimesh = parsed_data["mesh"]
        face_ids: np.ndarray = parsed_data["face_ids"]

        if len(mesh.faces) == 0:
            raise ValueError("Cannot sample points from an empty mesh.")

        # 1. Uniform Point Cloud Sampling
        uniform_points, uniform_face_indices = trimesh.sample.sample_surface(mesh, self.num_uniform)
        uniform_normals = mesh.face_normals[uniform_face_indices]
        uniform_face_ids = face_ids[uniform_face_indices] if len(face_ids) > 0 else uniform_face_indices

        # 2. Curvature / Feature-aware Point Cloud Sampling
        feature_points, feature_normals, feature_face_ids = self._sample_feature_aware(
            mesh, face_ids, self.num_feature
        )

        # Merge sampled sets
        all_points = np.vstack([uniform_points, feature_points])
        all_normals = np.vstack([uniform_normals, feature_normals])
        all_face_ids = np.concatenate([uniform_face_ids, feature_face_ids])

        return {
            "mesh": mesh,
            "point_cloud": all_points.astype(np.float32),
            "normals": all_normals.astype(np.float32),
            "point_face_ids": all_face_ids.astype(np.int32),
            "bounds": mesh.bounds.astype(np.float32),
            "surface_area": float(mesh.area),
            "volume": float(mesh.volume) if mesh.is_watertight else 0.0
        }

    def _sample_feature_aware(
        self, mesh: trimesh.Trimesh, face_ids: np.ndarray, count: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Weights face sampling probability by mean vertex angle variation (curvature proxy)."""
        face_angles = np.abs(mesh.face_adjacency_angles)
        face_weights = np.zeros(len(mesh.faces))

        for (f1, f2), angle in zip(mesh.face_adjacency, face_angles):
            face_weights[f1] += angle
            face_weights[f2] += angle

        total_weight = np.sum(face_weights)
        if total_weight > 0:
            probs = face_weights / total_weight
        else:
            probs = np.ones(len(mesh.faces)) / len(mesh.faces)

        chosen_faces = np.random.choice(len(mesh.faces), size=count, p=probs)
        
        # Sample random points inside chosen triangles
        triangles = mesh.triangles[chosen_faces]
        r1 = np.sqrt(np.random.rand(count, 1))
        r2 = np.random.rand(count, 1)

        a = 1 - r1
        b = r1 * (1 - r2)
        c = r1 * r2

        points = a * triangles[:, 0, :] + b * triangles[:, 1, :] + c * triangles[:, 2, :]
        normals = mesh.face_normals[chosen_faces]
        sampled_face_ids = face_ids[chosen_faces] if len(face_ids) > 0 else chosen_faces

        return points, normals, sampled_face_ids