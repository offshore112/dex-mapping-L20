"""Audit stage-1 personal-motion-to-R transfer and R-vs-L20 geometry.

Read-only analysis over existing v22/v23 outputs and the v12 R manifest.

Part A (stage 1) answers: with the current pipeline (theta_R = theta_i on a
fixed beta_R), how well are bone directions and normalized tip-pair
relationships preserved when the operator's motion is carried onto R?

Part B answers: how close is the constructed virtual hand R to L20 in bone
lengths, joint (semantic point) placement and palm width?  R's surface is a
standard MANO hand (beta_R = 0); only a global scale + rigid placement was
fitted to L20's 21 URDF-derived semantic points.

Both parts are diagnostics.  They do not validate the visual naturalness of
the source fit, contact/collision, L20 q feasibility or actuator commands.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_r_l20_feature_baseline_v15 import (
    CHAINS,
    POINT_ORDER,
    TIP_PAIRS,
    feature_arrays,
    make_feature_record,
)

BONE_NAMES = [
    f"{a}->{b}"
    for finger in CHAINS
    for a, b in zip(CHAINS[finger][:-1], CHAINS[finger][1:])
]
BONE_FINGERS = [finger for finger in CHAINS for _ in range(4)]


def finite_array(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}, got {array.shape}")
    return array


def load_sequence_points(path: Path):
    """Return source (operator, placed at R anchor) and R landmarks per frame."""
    data = json.loads(path.read_text(encoding="utf-8"))
    frames = data.get("frames") or data.get("sequence")
    if not frames:
        raise ValueError(f"{path} has no frames")
    source = finite_array(
        [f["source_reference_landmarks_m"] for f in frames], (len(frames), 21, 3), "source"
    )
    target = finite_array(
        [f["target_landmarks_m"] for f in frames], (len(frames), 21, 3), "target"
    )
    theta = finite_array([f["theta_hand_pose"] for f in frames], (len(frames), 45), "theta")
    ids = np.asarray([int(f["frame_id"]) for f in frames])
    times = finite_array([f["timestamp_s"] for f in frames], (len(frames),), "timestamps")
    return {
        "source": source,
        "target": target,
        "theta": theta,
        "frame_ids": ids,
        "timestamps": times,
        "metadata": {k: v for k, v in data.items() if k not in ("frames", "sequence")},
    }


def stage1_metrics(sequence: dict) -> dict:
    source, target = sequence["source"], sequence["target"]
    count = len(source)
    direction = np.zeros((count, len(BONE_NAMES)))
    tip_error = np.zeros((count, len(TIP_PAIRS)))
    tip_source = np.zeros((count, len(TIP_PAIRS)))
    tip_target = np.zeros((count, len(TIP_PAIRS)))
    palm_source = np.zeros(count)
    palm_target = np.zeros(count)
    point_distance = np.zeros((count, 21))

    for i in range(count):
        sf = feature_arrays(make_feature_record(source[i]))
        tf = feature_arrays(make_feature_record(target[i]))
        dot = np.clip(np.sum(sf["directions"] * tf["directions"], axis=1), -1.0, 1.0)
        direction[i] = np.degrees(np.arccos(dot))
        tip_error[i] = np.abs(sf["tip_distances"] - tf["tip_distances"])
        tip_source[i] = sf["tip_distances"]
        tip_target[i] = tf["tip_distances"]
        palm_source[i] = make_feature_record(source[i])["palm_scale_m"]
        palm_target[i] = make_feature_record(target[i])["palm_scale_m"]
        point_distance[i] = np.linalg.norm(source[i] - target[i], axis=1)

    per_bone = {}
    for j, name in enumerate(BONE_NAMES):
        column = direction[:, j]
        per_bone[name] = {
            "finger": BONE_FINGERS[j],
            "mean_deg": float(column.mean()),
            "p95_deg": float(np.percentile(column, 95)),
            "max_deg": float(column.max()),
            "worst_frame_id": int(sequence["frame_ids"][int(np.argmax(column))]),
        }
    per_finger = {}
    for finger in CHAINS:
        columns = [j for j in range(len(BONE_NAMES)) if BONE_FINGERS[j] == finger]
        per_finger[finger] = {
            "bone_direction_mean_deg": float(direction[:, columns].mean()),
            "bone_direction_rmse_deg": float(np.sqrt(np.mean(direction[:, columns] ** 2))),
            "bone_direction_max_deg": float(direction[:, columns].max()),
        }

    per_pair = {}
    for j, pair in enumerate(TIP_PAIRS):
        key = "+".join(pair)
        nearest = int(np.argmin(tip_source[:, j]))
        per_pair[key] = {
            "mean_abs_normalized_error": float(tip_error[:, j].mean()),
            "max_abs_normalized_error": float(tip_error[:, j].max()),
            "source_closest_frame_id": int(sequence["frame_ids"][nearest]),
            "at_source_closest": {
                "source_distance_normalized": float(tip_source[nearest, j]),
                "target_distance_normalized": float(tip_target[nearest, j]),
                "source_distance_mm": float(
                    tip_source[nearest, j] * palm_source[nearest] * 1000.0
                ),
                "target_distance_mm": float(
                    tip_target[nearest, j] * palm_target[nearest] * 1000.0
                ),
            },
        }

    theta = sequence["theta"]
    pose_step = np.abs(np.diff(theta, axis=0)).max() if count > 1 else 0.0

    return {
        "frame_count": int(count),
        "duration_s": float(sequence["timestamps"][-1] - sequence["timestamps"][0]),
        "transfer_definition": "theta_R(t) = theta_i(t); fixed beta_R; wrist-anchored",
        "pose_parameter_transfer_error_rad": 0.0,
        "palm_scale_source_m": float(palm_source.mean()),
        "palm_scale_target_m": float(palm_target.mean()),
        "palm_scale_ratio_target_over_source": float(palm_target.mean() / palm_source.mean()),
        "bone_direction_rmse_deg": float(np.sqrt(np.mean(direction ** 2))),
        "bone_direction_mean_deg": float(direction.mean()),
        "bone_direction_p95_deg": float(np.percentile(direction, 95)),
        "bone_direction_max_deg": float(direction.max()),
        "per_finger": per_finger,
        "per_bone": per_bone,
        "tip_pair_mean_abs_normalized_error": float(tip_error.mean()),
        "tip_pair_max_abs_normalized_error": float(tip_error.max()),
        "tip_pairs": per_pair,
        "point_position_difference_mm": {
            "mean": float(point_distance.mean() * 1000.0),
            "max": float(point_distance.max() * 1000.0),
            "note": "absolute source-vs-R distance; dominated by the size ratio, diagnostic only",
        },
        "pose_step_max_abs_rad": float(pose_step),
    }


def geometry_metrics(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    arrays_path = manifest_path.parent / manifest["arrays"]["npz"]
    with np.load(arrays_path, allow_pickle=False) as arrays:
        l20 = finite_array(arrays["target_landmarks_m"], (21, 3), "L20 points")
        r_points = finite_array(arrays["fitted_landmarks_m"], (21, 3), "R points")
        point_error = finite_array(arrays["point_error_m"], (21,), "point error")
        theta_ref = finite_array(arrays["theta_ref_hand_pose"], (45,), "theta_ref")
        beta_R = finite_array(arrays["beta_R"], (10,), "beta_R")
        scale = float(arrays["scale"])

    index = {name: i for i, name in enumerate(POINT_ORDER)}

    def bone_lengths(points):
        out = {}
        for finger, chain in CHAINS.items():
            for a, b in zip(chain[:-1], chain[1:]):
                out[f"{a}->{b}"] = float(np.linalg.norm(points[index[b]] - points[index[a]]))
        return out

    l20_bones, r_bones = bone_lengths(l20), bone_lengths(r_points)
    bone_rows = []
    for name in BONE_NAMES:
        l_val, r_val = l20_bones[name], r_bones[name]
        bone_rows.append({
            "bone": name,
            "finger": BONE_FINGERS[BONE_NAMES.index(name)],
            "l20_mm": l_val * 1000.0,
            "r_mm": r_val * 1000.0,
            "difference_mm": (r_val - l_val) * 1000.0,
            "ratio_r_over_l20": r_val / l_val,
        })

    l20_lengths = np.asarray([row["l20_mm"] for row in bone_rows])
    r_lengths = np.asarray([row["r_mm"] for row in bone_rows])
    abs_diff = np.abs(l20_lengths - r_lengths)

    finger_totals = {}
    for finger in CHAINS:
        rows = [row for row in bone_rows if row["finger"] == finger]
        l_total = sum(row["l20_mm"] for row in rows[1:])  # exclude wrist->mcp
        r_total = sum(row["r_mm"] for row in rows[1:])
        finger_totals[finger] = {
            "l20_mm": l_total,
            "r_mm": r_total,
            "difference_mm": r_total - l_total,
            "ratio_r_over_l20": r_total / l_total,
        }

    def span(points, a, b):
        return float(np.linalg.norm(points[index[a]] - points[index[b]]))

    palm = {
        "width_index_mcp_to_pinky_mcp_mm": {
            "l20": span(l20, "index_mcp", "pinky_mcp") * 1000.0,
            "r": span(r_points, "index_mcp", "pinky_mcp") * 1000.0,
        },
        "length_wrist_to_middle_mcp_mm": {
            "l20": span(l20, "wrist", "middle_mcp") * 1000.0,
            "r": span(r_points, "wrist", "middle_mcp") * 1000.0,
        },
        "width_index_mcp_to_ring_mcp_mm": {
            "l20": span(l20, "index_mcp", "ring_mcp") * 1000.0,
            "r": span(r_points, "index_mcp", "ring_mcp") * 1000.0,
        },
        "thumb_cmc_to_pinky_mcp_mm": {
            "l20": span(l20, "thumb_cmc", "pinky_mcp") * 1000.0,
            "r": span(r_points, "thumb_cmc", "pinky_mcp") * 1000.0,
        },
    }
    for key, value in palm.items():
        value["difference_mm"] = value["r"] - value["l20"]
        value["ratio_r_over_l20"] = value["r"] / value["l20"]

    per_point = {
        name: {
            "l20_distance_from_wrist_mm": float(
                np.linalg.norm(l20[index[name]] - l20[index["wrist"]]) * 1000.0
            ),
            "r_distance_from_wrist_mm": float(
                np.linalg.norm(r_points[index[name]] - r_points[index["wrist"]]) * 1000.0
            ),
            "point_distance_mm": float(point_error[index[name]] * 1000.0),
        }
        for name in POINT_ORDER
    }

    return {
        "model_manifest": str(manifest_path),
        "placement": {
            "beta_R_norm": float(np.linalg.norm(beta_R)),
            "theta_ref_hand_pose_norm": float(np.linalg.norm(theta_ref)),
            "scale": scale,
            "note": "beta_R=0 and theta_ref=0 means R is a standard flat MANO hand; "
                    "only global scale and rigid placement were fitted to L20 points",
        },
        "bone_lengths": bone_rows,
        "bone_length_abs_error_mm": {
            "mean": float(abs_diff.mean()),
            "max": float(abs_diff.max()),
            "max_bone": bone_rows[int(np.argmax(abs_diff))]["bone"],
            "mean_abs_ratio_error": float(np.abs(r_lengths / l20_lengths - 1.0).mean()),
        },
        "finger_totals": finger_totals,
        "palm": palm,
        "point_position": per_point,
        "point_position_rmse_mm": float(np.sqrt(np.mean(point_error ** 2)) * 1000.0),
        "point_position_max_mm": float(point_error.max() * 1000.0),
        "point_position_definition": "semantic point distance in the fitted placement frame; "
                                     "matches the v12 static-fit report",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence", type=Path,
        default=Path("experiment/v23_pinky_distal_twist/run2_personal_mapping_test/"
                     "fixed_r/r_sequence_v22.json"),
    )
    parser.add_argument(
        "--baseline", type=Path,
        default=Path("experiment/v22_personal_fixed_r/r_sequence_v22.json"),
    )
    parser.add_argument(
        "--model", type=Path,
        default=Path("experiment/v12_R_L20_canonical_human/r_l20_model.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("experiment/v24_stage1_R_L20_audit"),
    )
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.json"
    if metrics_path.exists():
        raise SystemExit(f"{metrics_path} exists; use a new output directory")

    print(json.dumps({"status": "loading", "sequence": str(args.sequence)}), flush=True)
    current = load_sequence_points(args.sequence.resolve())
    stage1 = stage1_metrics(current)

    baseline = None
    if args.baseline and args.baseline.exists():
        print(json.dumps({"status": "baseline", "sequence": str(args.baseline)}), flush=True)
        baseline = stage1_metrics(load_sequence_points(args.baseline.resolve()))

    print(json.dumps({"status": "geometry", "model": str(args.model)}), flush=True)
    geometry = geometry_metrics(args.model.resolve())

    result = {
        "version": "v24_stage1_R_L20_audit",
        "status": "completed_readonly_diagnostic",
        "inputs": {
            "sequence": str(args.sequence),
            "baseline_sequence": str(args.baseline) if args.baseline else None,
            "model": str(args.model),
        },
        "stage1_mapping": stage1,
        "stage1_mapping_baseline_v22": baseline,
        "r_vs_l20_geometry": geometry,
        "limitations": [
            "stage-1 pose transfer is theta_R = theta_i, so pose parameters are identical by construction",
            "residual differences measure morphology, not a learned cross-size mapping",
            "source fitting quality is not validated here; v23 action naturalness failed user review",
            "R surface is a standard MANO hand (beta_R = 0) and does not match L20 surface/CAD",
            "no L20 q solving, collision, contact or actuator u in this audit",
        ],
    }
    metrics_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "config.json").write_text(
        json.dumps({"sequence": str(args.sequence), "baseline": str(args.baseline),
                    "model": str(args.model)}, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "done",
        "stage1_bone_direction_rmse_deg": stage1["bone_direction_rmse_deg"],
        "stage1_tip_pair_mae": stage1["tip_pair_mean_abs_normalized_error"],
        "bone_length_mean_abs_error_mm": geometry["bone_length_abs_error_mm"]["mean"],
        "point_position_rmse_mm": geometry["point_position_rmse_mm"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
