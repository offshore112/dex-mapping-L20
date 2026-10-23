"""Load the shared MANO-R/L20 target hand model and evaluate its L20 landmarks."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from l20_kinematics import HandURDF, transform_points  # noqa: E402


class SharedRL20:
    """Shared target hand: fixed MANO-R geometry plus L20 kinematics.

    The model deliberately keeps the geometry and robot state separate:
    ``vertices_m`` is the fixed reference MANO-R surface, while ``landmarks``
    evaluates L20 semantic points from independent joint coordinates ``q``.
    This class does not infer actuator commands ``u``.
    """

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        array_path = self.manifest_path.parent / self.manifest["arrays"]["npz"]
        self.arrays = np.load(array_path, allow_pickle=False)
        self.vertices_m = np.asarray(self.arrays["mano_vertices_ref_m"], dtype=float)
        self.faces = np.asarray(self.arrays["mano_faces"], dtype=int)
        self.target_landmarks_m = np.asarray(self.arrays["target_landmarks_m"], dtype=float)
        self.fitted_landmarks_m = np.asarray(self.arrays["fitted_landmarks_m"], dtype=float)
        self.point_error_m = np.asarray(self.arrays["point_error_m"], dtype=float)
        urdf_path = Path(self.manifest["sources"]["urdf"])
        if not urdf_path.is_file():
            raise FileNotFoundError(f"L20 URDF not found: {urdf_path}")
        self.robot = HandURDF(urdf_path)
        self.point_sources = self.manifest["kinematics"]["point_sources"]
        self.q_order = list(self.manifest["kinematics"]["independent_q_order"])

    def _q_dict(self, q: Mapping[str, float] | np.ndarray | None) -> dict[str, float]:
        if q is None:
            return {}
        if isinstance(q, Mapping):
            return {str(k): float(v) for k, v in q.items()}
        values = np.asarray(q, dtype=float).reshape(-1)
        if values.size != len(self.q_order):
            raise ValueError(f"q expects {len(self.q_order)} independent values, got {values.size}")
        return dict(zip(self.q_order, values.tolist()))

    def landmarks(self, q: Mapping[str, float] | np.ndarray | None = None) -> np.ndarray:
        """Return the 21 L20 semantic points in the hand_base_link frame."""
        transforms, frames, _ = self.robot.fk(self._q_dict(q))
        points = []
        for source in self.point_sources:
            kind = source["kind"]
            local = np.asarray(source.get("local_xyz_m", [0, 0, 0]), dtype=float)
            if kind == "joint_frame":
                points.append(frames[source["joint"]][:3, 3])
            elif kind == "link_local":
                points.append(transform_points(transforms[source["link"]], local[None])[0])
            elif kind == "root_local":
                points.append(transform_points(transforms[self.robot.root], local[None])[0])
            else:
                raise ValueError(f"Unknown point source kind: {kind}")
        return np.asarray(points, dtype=float)

    def expand_q(self, q: Mapping[str, float] | np.ndarray | None = None) -> dict[str, float]:
        """Expand independent q to all 21 URDF joint coordinates, including mimic."""
        return self.robot.expand(self._q_dict(q))

    def zero_pose_residual_m(self) -> np.ndarray:
        return self.landmarks() - self.target_landmarks_m

