"""
Stage 04: Validate - Geometric validation, manifold integrity check, and deduplication hashing.
"""

import hashlib
from typing import Dict, Any, Tuple, Optional
import numpy as np


class GeometryValidator:
    def __init__(
        self, 
        require_watertight: bool = True, 
        min_volume: float = 1e-6, 
        max_bounding_box_diag: float = 1000.0,
        max_bbox_diag: Optional[float] = None
    ):
        self.require_watertight = require_watertight
        self.min_volume = min_volume
        self.max_bounding_box_diag = max_bbox_diag if max_bbox_diag is not None else max_bounding_box_diag
        self.seen_hashes = set()

    def compute_geometric_hash(self, point_cloud: np.ndarray, bins: int = 64) -> str:
        """Computes spatial-histogram invariant hash for deduplication."""
        min_p, max_p = point_cloud.min(axis=0), point_cloud.max(axis=0)
        norm_pc = (point_cloud - min_p) / (max_p - min_p + 1e-8)
        hist, _ = np.histogramdd(norm_pc, bins=bins, range=[(0, 1), (0, 1), (0, 1)])
        return hashlib.md5(hist.tobytes()).hexdigest()

    def validate(self, converted_data: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[str]]:
        mesh = converted_data["mesh"]
        point_cloud = converted_data["point_cloud"]

        # Check 1: Non-empty vertices/faces
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            return False, "EMPTY_GEOMETRY", "Mesh contains zero vertices or faces."

        # Check 2: Bounding Box scaling
        bbox_diag = np.linalg.norm(converted_data["bounds"][1] - converted_data["bounds"][0])
        if bbox_diag > self.max_bounding_box_diag:
            return False, "INVALID_BOUNDS", f"Bounding box diagonal ({bbox_diag:.2f}) exceeds maximum ({self.max_bounding_box_diag})."

        # Check 3: Watertightness check
        if self.require_watertight and not mesh.is_watertight:
            return False, "NON_WATERTIGHT", "Mesh surface is open/non-watertight."

        # Check 4: Non-zero Volume check
        if self.require_watertight and converted_data["volume"] < self.min_volume:
            return False, "DEGENERATE_VOLUME", f"Volume ({converted_data['volume']:.8f}) below threshold."

        # Check 5: Geometric Deduplication
        geo_hash = self.compute_geometric_hash(point_cloud)
        if geo_hash in self.seen_hashes:
            return False, "DUPLICATE_GEOMETRY", "Identical geometric signature already registered."
        
        self.seen_hashes.add(geo_hash)
        return True, None, geo_hash