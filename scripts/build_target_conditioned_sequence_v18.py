"""Build a fixed L20-conditioned functional skeleton target sequence.

The natural MANO surface remains the display/source representation.  This
adapter calibrates one target-conditioned skeleton at the open reference:
L20 zero-pose bone lengths, palm frame and directions are fixed; each later
MANO bone direction is transferred as a relative rotation from its MANO open
direction onto the corresponding L20 reference direction.  It is a single
calibration, not per-frame point fitting.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_r_l20_feature_baseline_v15 import CHAINS, POINT_ORDER, make_feature_record  # noqa: E402
from shared_r_l20 import SharedRL20  # noqa: E402


def minimal_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return a 3x3 column-vector rotation that maps source to target."""
    a = np.asarray(source, dtype=float)
    b = np.asarray(target, dtype=float)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    cross = np.cross(a, b)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if sine < 1e-10:
        if cosine > 0.0:
            return np.eye(3)
        axis = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
        axis -= np.dot(axis, a) * a
        axis /= np.linalg.norm(axis)
        skew = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
        return np.eye(3) + 2.0 * (skew @ skew)
    axis = cross / sine
    skew = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + skew * sine + (skew @ skew) * (1.0 - cosine)


def edge_records(record: dict) -> dict[str, dict]:
    return {edge["name"]: edge for edge in record["edges"]}


def adapt_frame(source_points: np.ndarray, source_reference: dict, target_reference: dict,
                target_wrist_world_m: np.ndarray) -> tuple[np.ndarray, dict]:
    source = make_feature_record(source_points)
    source_ref_edges = edge_records(source_reference)
    source_edges = edge_records(source)
    target_edges = edge_records(target_reference)
    adapted_local = np.zeros((len(POINT_ORDER), 3), dtype=float)
    point_index = {name: i for i, name in enumerate(POINT_ORDER)}
    for chain in CHAINS.values():
        adapted_local[point_index[chain[0]]] = [0.0, 0.0, 0.0]
        for first, second in zip(chain[:-1], chain[1:]):
            edge_name = f"{first}->{second}"
            ref_dir = np.asarray(source_ref_edges[edge_name]["direction"], dtype=float)
            current_dir = np.asarray(source_edges[edge_name]["direction"], dtype=float)
            l20_dir = np.asarray(target_edges[edge_name]["direction"], dtype=float)
            direction_rotation = minimal_rotation(ref_dir, current_dir)
            adapted_dir = direction_rotation @ l20_dir
            adapted_dir /= np.linalg.norm(adapted_dir)
            length = float(target_edges[edge_name]["length_normalized"])
            adapted_local[point_index[second]] = adapted_local[point_index[first]] + length * adapted_dir
    target_basis = np.asarray(target_reference["palm_basis_root_columns"], dtype=float)
    target_scale = float(target_reference["palm_scale_m"])
    target_wrist = np.asarray(target_wrist_world_m, dtype=float)
    adapted_world = (adapted_local * target_scale) @ target_basis.T + target_wrist
    diagnostics = {
        "source_palm_scale_m": float(source["palm_scale_m"]),
        "target_palm_scale_m": target_scale,
        "adapted_points_normalized": adapted_local.tolist(),
        "direction_transfer": "minimal rotation from MANO reference edge direction to current edge direction, applied to L20 reference edge direction",
    }
    return adapted_world, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = SharedRL20(args.model.resolve())
    if model.manifest["kinematics"]["point_order"] != POINT_ORDER:
        raise ValueError("Model point order differs from frozen v18 semantic order")
    sequence_payload = json.loads(args.sequence.resolve().read_text(encoding="utf-8"))
    frames = sequence_payload.get("sequence", sequence_payload.get("frames"))
    if not isinstance(frames, list) or not frames:
        raise ValueError("Input must contain a non-empty sequence or frames list")
    source_reference = make_feature_record(np.asarray(frames[0]["target_landmarks_m"], dtype=float))
    zero_q = np.zeros(len(model.q_order), dtype=float)
    target_reference_points = model.landmarks(zero_q)
    target_reference = make_feature_record(target_reference_points)
    adapted_frames = []
    diagnostics = []
    for frame in frames:
        adapted, info = adapt_frame(np.asarray(frame["target_landmarks_m"], dtype=float), source_reference,
                                    target_reference, target_reference_points[0])
        adapted_frame = dict(frame)
        adapted_frame["source_target_landmarks_m"] = frame["target_landmarks_m"]
        # Preserve the original R task geometry for selected fingertip pairs;
        # the L20-conditioned skeleton is used for shape regularization only.
        adapted_frame["task_target_landmarks_m"] = frame["target_landmarks_m"]
        adapted_frame["target_landmarks_m"] = adapted.tolist()
        adapted_frame["adapter"] = "L20_zero_reference_relative_direction_transfer"
        adapted_frames.append(adapted_frame)
        diagnostics.append({"frame_id": frame["frame_id"], **info})
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "v18_L20_target_conditioned_skeleton_sequence",
        "status": "built_fixed_target_conditioned_skeleton_targets",
        "model": str(args.model.resolve()),
        "source_sequence": str(args.sequence.resolve()),
        "reference_frame_id": int(frames[0]["frame_id"]),
        "adapter": {
            "type": "fixed_target_conditioned_functional_skeleton",
            "target_reference": "L20 URDF zero-pose semantic points, represented in its own palm-normalized frame",
            "bone_lengths": "fixed to L20 reference bone lengths",
            "motion": "relative MANO edge-direction rotation transferred onto L20 reference directions",
            "per_frame_absolute_fit": False,
        },
        "sequence": adapted_frames,
        "diagnostics": diagnostics,
        "limitations": [
            "The open frame is a fixed simulation reference, not hardware calibration.",
            "The adapter changes the functional skeleton target, not the natural MANO surface.",
            "The sequence remains synthetic interpolation, not camera data.",
        ],
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "frame_count": len(adapted_frames), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
