"""Run the v15 functional-feature R->L20 q baseline.

The v14 baseline compared 21 absolute MANO/L20 proxy points.  v15 keeps the
MANO surface as a human-looking display model, but compares root-relative,
palm-normalized directions, lengths, palm spread and fingertip task features.
The optimization variable is still the 16-dimensional independent L20 q; the
five URDF mimic joints are expanded by the existing kinematics implementation.
This script does not infer actuator commands u or claim contact success.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared_r_l20 import SharedRL20  # noqa: E402


POINT_ORDER = [
    "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]

CHAINS = {
    "thumb": ["wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip"],
    "index": ["wrist", "index_mcp", "index_pip", "index_dip", "index_tip"],
    "middle": ["wrist", "middle_mcp", "middle_pip", "middle_dip", "middle_tip"],
    "ring": ["wrist", "ring_mcp", "ring_pip", "ring_dip", "ring_tip"],
    "pinky": ["wrist", "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"],
}
MCP_NAMES = ["thumb_cmc", "index_mcp", "middle_mcp", "ring_mcp", "pinky_mcp"]
TIP_NAMES = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]
TIP_PAIRS = list(itertools.combinations(TIP_NAMES, 2))


def unit(vector: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(value))
    if norm < 1e-12:
        raise ValueError(f"Degenerate vector while building palm frame: {name}")
    return value / norm


def make_feature_record(points: np.ndarray) -> dict:
    """Build a morphology-normalized feature record from 21 semantic points."""
    points = np.asarray(points, dtype=float)
    if points.shape != (21, 3):
        raise ValueError(f"Expected 21x3 points, got {points.shape}")
    if POINT_ORDER != [name for name in POINT_ORDER]:
        raise AssertionError("Point order must remain explicit")
    index = {name: i for i, name in enumerate(POINT_ORDER)}
    wrist = points[index["wrist"]]

    # Build a palm frame independently for R and L20.  z points from the
    # wrist to the middle MCP; x spans the index/pinky MCPs; y completes the
    # right-handed orthonormal frame.  Absolute root placement and orientation
    # are therefore not part of the feature objective.
    z = unit(points[index["middle_mcp"]] - wrist, "wrist->middle_mcp")
    x_raw = points[index["index_mcp"]] - points[index["pinky_mcp"]]
    x = unit(x_raw - np.dot(x_raw, z) * z, "pinky_mcp->index_mcp projected")
    y = unit(np.cross(z, x), "palm normal")
    basis = np.column_stack((x, y, z))
    palm_scale = float(np.linalg.norm(points[index["middle_mcp"]] - wrist))
    if palm_scale < 1e-12:
        raise ValueError("Degenerate palm scale")

    local = (points - wrist) @ basis / palm_scale
    edges = []
    for finger, chain in CHAINS.items():
        for a, b in zip(chain[:-1], chain[1:]):
            vector = (points[index[b]] - points[index[a]]) @ basis
            length_m = float(np.linalg.norm(vector))
            if length_m < 1e-12:
                raise ValueError(f"Degenerate edge {a}->{b}")
            edges.append({
                "name": f"{a}->{b}",
                "finger": finger,
                "start": a,
                "end": b,
                "direction": (vector / length_m).tolist(),
                "length_normalized": length_m / palm_scale,
            })

    tip_pairs = []
    for first, second in TIP_PAIRS:
        vector = local[index[second]] - local[index[first]]
        distance = float(np.linalg.norm(vector))
        direction = (vector / distance).tolist() if distance >= 1e-12 else [0.0, 0.0, 0.0]
        tip_pairs.append({
            "pair": [first, second],
            "distance_normalized": distance,
            "direction": direction,
        })

    selected_pairs = {"+".join(item["pair"]): item for item in tip_pairs}
    return {
        "palm_scale_m": palm_scale,
        "palm_basis_root_columns": basis.tolist(),
        "canonical_points": local.tolist(),
        "mcp_points": [local[index[name]].tolist() for name in MCP_NAMES],
        "edges": edges,
        "tip_pairs": tip_pairs,
        "tip_pair_lookup": selected_pairs,
    }


def feature_arrays(record: dict) -> dict:
    return {
        "directions": np.asarray([edge["direction"] for edge in record["edges"]], dtype=float),
        "lengths": np.asarray([edge["length_normalized"] for edge in record["edges"]], dtype=float),
        "mcp_points": np.asarray(record["mcp_points"], dtype=float),
        "tip_distances": np.asarray([item["distance_normalized"] for item in record["tip_pairs"]], dtype=float),
        "tip_directions": np.asarray([item["direction"] for item in record["tip_pairs"]], dtype=float),
    }


def selected_pair_index(pair: list[str] | None) -> int | None:
    if not pair:
        return None
    key = "+".join(pair)
    for i, item in enumerate(TIP_PAIRS):
        if "+".join(item) == key:
            return i
    raise ValueError(f"Unknown fingertip pair: {pair}")


def residual_for(q: np.ndarray, model: SharedRL20, target: dict, weights: dict) -> np.ndarray:
    actual = feature_arrays(make_feature_record(model.landmarks(q)))
    desired = feature_arrays(target)
    pieces = [
        (actual["directions"] - desired["directions"]).reshape(-1) * weights["direction"],
        (actual["lengths"] - desired["lengths"]).reshape(-1) * weights["length"],
        (actual["mcp_points"] - desired["mcp_points"]).reshape(-1) * weights["mcp"],
        (actual["tip_distances"] - desired["tip_distances"]).reshape(-1) * weights["pair_distance"],
    ]
    # A distance close to zero has no stable direction.  Do not inject an
    # arbitrary direction target for a nearly coincident pinch pair.
    pieces.append((actual["tip_directions"] - desired["tip_directions"]).reshape(-1) * weights["pair_direction"])
    pair_index = selected_pair_index(weights.get("selected_pair"))
    if pair_index is not None:
        pieces.append(np.asarray([
            (actual["tip_distances"][pair_index] - desired["tip_distances"][pair_index])
            * weights["selected_pair_distance"]
        ]))
        desired_distance = desired["tip_distances"][pair_index]
        if desired_distance >= weights["pair_direction_min_distance"]:
            pieces.append((actual["tip_directions"][pair_index] - desired["tip_directions"][pair_index])
                          * weights["selected_pair_direction"])
    return np.concatenate(pieces)


def mimic_residual(model: SharedRL20, q: np.ndarray) -> float:
    full = model.expand_q(q)
    values = []
    for joint in model.manifest["kinematics"]["joints"]:
        if joint["mimic"] is not None:
            m = joint["mimic"]
            values.append(full[joint["name"]] - (m["multiplier"] * full[m["joint"]] + m["offset"]))
    return float(max(np.abs(values), default=0.0))


def solve_action(model: SharedRL20, action: dict, starts: list[np.ndarray], weights: dict) -> dict:
    from scipy.optimize import least_squares

    target_points = np.asarray(action["target_landmarks_m"], dtype=float)
    target_record = make_feature_record(target_points)
    bounds = np.asarray([model.robot.bounds[name] for name in model.q_order], dtype=float)
    trials = []
    for start_id, start in enumerate(starts):
        result = least_squares(
            lambda q: residual_for(q, model, target_record, weights),
            np.clip(start, bounds[:, 0], bounds[:, 1]),
            bounds=(bounds[:, 0], bounds[:, 1]), method="trf",
            xtol=1e-11, ftol=1e-11, gtol=1e-11, max_nfev=900,
            diff_step=1e-5,
        )
        actual_points = model.landmarks(result.x)
        actual_record = make_feature_record(actual_points)
        actual_features = feature_arrays(actual_record)
        target_features = feature_arrays(target_record)
        trials.append({
            "start_id": start_id,
            "success": bool(result.success),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "cost": float(result.cost),
            "q": result.x,
            "actual_record": actual_record,
            "actual_features": actual_features,
            "target_features": target_features,
            "target_record": target_record,
        })
    best = min(trials, key=lambda item: item["cost"])
    actual = best["actual_features"]
    target = best["target_features"]
    angle_errors = np.degrees(np.arccos(np.clip(np.sum(actual["directions"] * target["directions"], axis=1), -1.0, 1.0)))
    direction_errors = np.degrees(np.arccos(np.clip(np.sum(actual["tip_directions"] * target["tip_directions"], axis=1), -1.0, 1.0)))
    selected = action.get("tip_pair")
    selected_i = selected_pair_index(selected)
    q = best["q"]
    absolute_error = model.landmarks(q) - target_points
    pair_info = None
    if selected_i is not None:
        target_dist_mm = float(target["tip_distances"][selected_i] * best["target_record"]["palm_scale_m"] * 1000.0)
        actual_dist_mm = float(actual["tip_distances"][selected_i] * best["actual_record"]["palm_scale_m"] * 1000.0)
        pair_info = {
            "pair": selected,
            "target_distance_normalized": float(target["tip_distances"][selected_i]),
            "actual_distance_normalized": float(actual["tip_distances"][selected_i]),
            "normalized_distance_error": float(actual["tip_distances"][selected_i] - target["tip_distances"][selected_i]),
            "target_distance_mm_in_R_input": target_dist_mm,
            "actual_distance_mm_in_L20_root": actual_dist_mm,
            "direction_error_deg": float(direction_errors[selected_i]),
            "direction_used_in_objective": bool(target["tip_distances"][selected_i] >= weights["pair_direction_min_distance"]),
        }
    return {
        "name": action["name"],
        "kind": action.get("kind", "synthetic_R_target"),
        "tip_pair": selected,
        "solver": {
            "multi_start_count": len(starts),
            "best_start_id": int(best["start_id"]),
            "success": bool(best["success"]),
            "message": best["message"],
            "nfev": best["nfev"],
            "cost": best["cost"],
            "feature_weights": {key: value for key, value in weights.items()},
        },
        "q_order": model.q_order,
        "q_independent_rad": q.tolist(),
        "q_full_rad": model.expand_q(q),
        "feature_metrics": {
            "bone_direction_rmse_deg": float(np.sqrt(np.mean(angle_errors ** 2))),
            "bone_direction_mean_deg": float(np.mean(angle_errors)),
            "bone_direction_max_deg": float(np.max(angle_errors)),
            "normalized_bone_length_rmse": float(np.sqrt(np.mean((actual["lengths"] - target["lengths"]) ** 2))),
            "normalized_mcp_position_rmse": float(np.sqrt(np.mean((actual["mcp_points"] - target["mcp_points"]) ** 2))),
            "normalized_all_tip_pair_distance_rmse": float(np.sqrt(np.mean((actual["tip_distances"] - target["tip_distances"]) ** 2))),
            "tip_pair_direction_rmse_deg": float(np.sqrt(np.mean(direction_errors ** 2))),
        },
        "selected_pair_metrics": pair_info,
        "absolute_landmark_diagnostic": {
            "rmse_mm": float(np.sqrt(np.mean(absolute_error ** 2)) * 1000.0),
            "max_mm": float(np.max(np.linalg.norm(absolute_error, axis=1)) * 1000.0),
            "note": "diagnostic only; absolute MANO/L20 proxy points are not the v15 objective",
        },
        "limit_violations": 0,
        "mimic_residual_max_rad": mimic_residual(model, q),
        "collision_contact": "not_evaluated",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="shared R/L20 manifest; v12 natural surface is the current baseline")
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = SharedRL20(args.model.resolve())
    if model.manifest["kinematics"]["point_order"] != POINT_ORDER:
        raise ValueError("Model point order differs from the frozen v15 semantic order")
    actions_payload = json.loads(args.actions.resolve().read_text(encoding="utf-8"))
    bounds = np.asarray([model.robot.bounds[name] for name in model.q_order], dtype=float)
    rng = np.random.default_rng(20261017)
    starts = [
        np.zeros(len(model.q_order), dtype=float),
        (bounds[:, 0] + bounds[:, 1]) / 2.0,
        bounds[:, 0] + 0.25 * (bounds[:, 1] - bounds[:, 0]),
        rng.uniform(bounds[:, 0], bounds[:, 1]),
    ]
    base_weights = {
        "direction": 1.0,
        "length": 0.65,
        "mcp": 0.75,
        "pair_distance": 0.70,
        "pair_direction": 0.45,
        "selected_pair_distance": 2.5,
        "selected_pair_direction": 1.25,
        "pair_direction_min_distance": 0.025,
        "selected_pair": None,
    }
    results = []
    feature_targets = []
    for action in actions_payload["actions"]:
        weights = dict(base_weights)
        weights["selected_pair"] = action.get("tip_pair")
        if action.get("tip_pair"):
            # Task-priority mode: pinch/opposition must first preserve the
            # selected fingertip relation.  Other whole-hand features remain
            # as regularizers, rather than preventing a reachable task.
            weights.update({
                "direction": 0.50,
                "length": 0.20,
                "mcp": 0.20,
                "pair_distance": 0.15,
                "pair_direction": 0.10,
                "selected_pair_distance": 30.0,
                "selected_pair_direction": 0.20,
            })
        target_record = make_feature_record(np.asarray(action["target_landmarks_m"], dtype=float))
        feature_targets.append({"name": action["name"], "features": target_record})
        results.append(solve_action(model, action, starts, weights))

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    model_record = {
        "version": "v15_R_L20_functional_feature_model",
        "status": "constructed_and_evaluated_synthetic_baseline",
        "representation": {
            "surface": "natural MANO reference surface from the supplied shared model; surface is not a L20 shell",
            "skeleton": "human semantic wrist/CMC-MCP-PIP-DIP/tip points represented by relative features",
            "adapter": "L20 URDF independent q with limits and mimic expansion",
        },
        "surface_source_manifest": str(args.model.resolve()),
        "point_order": POINT_ORDER,
        "chains": CHAINS,
        "normalization": {
            "root": "wrist",
            "palm_frame": "z=wrist->middle_mcp; x=pinky_mcp->index_mcp projected; y=cross(z,x)",
            "scale": "distance wrist->middle_mcp per frame",
            "absolute_point_matching": False,
        },
        "feature_weights": base_weights,
        "q_space": {"order": model.q_order, "count": len(model.q_order), "mimic_count": 5},
        "u_space": "not_defined_without_E_L20 actuator descriptor",
    }
    (out / "r_l20_v15_model.json").write_text(json.dumps(model_record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "feature_targets.json").write_text(json.dumps({
        "version": "v15_R_feature_targets",
        "source_actions": str(args.actions.resolve()),
        "targets": feature_targets,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    payload = {
        "version": "v15_R_L20_functional_feature_baseline",
        "status": "completed_synthetic_feature_level_R_to_L20_q_baseline",
        "model": str(args.model.resolve()),
        "actions": str(args.actions.resolve()),
        "optimizer": {
            "method": "scipy.least_squares bounded multi-start",
            "objective": "root-relative palm-normalized directions, bone lengths, MCP spread, fingertip pair distances/directions",
            "starts": len(starts),
            "q_space": "16 independent L20 coordinates; 5 mimic joints expanded by URDF",
            "u_space": "not defined; actuator descriptor unavailable",
        },
        "results": results,
        "limitations": [
            "Targets are fixed synthetic MANO pose vectors, not camera estimates or a dataset.",
            "Absolute MANO/L20 proxy point error is retained only as a diagnostic, not the v15 objective.",
            "Collision, contact normal, object geometry, temporal smoothness and actuator u were not evaluated in this first feature baseline.",
            "Near-zero fingertip distance alone is not called successful pinch contact.",
        ],
    }
    (out / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# v15：R→L20 功能特征基线",
        "",
        "本版本使用自然 MANO 表面对应的语义骨架特征，不再把 MANO 21 个绝对点逐点当作 L20 的绝对目标。特征在各自掌部坐标系中计算，并以腕部到中指 MCP 距离归一化。优化变量为 L20 的 16 个独立 q，5 条 mimic 由 URDF 展开。",
        "",
        "## 结果",
        "",
        "| action | 骨方向 RMSE (deg) | 归一化骨长 RMSE | MCP spread RMSE | 选定指尖对归一化距离误差 | L20 mimic residual (rad) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        pair = result["selected_pair_metrics"]
        pair_error = "—" if pair is None else f"{pair['normalized_distance_error']:.4f}"
        lines.append(
            f"| {result['name']} | {result['feature_metrics']['bone_direction_rmse_deg']:.3f} | "
            f"{result['feature_metrics']['normalized_bone_length_rmse']:.4f} | "
            f"{result['feature_metrics']['normalized_mcp_position_rmse']:.4f} | {pair_error} | "
            f"{result['mimic_residual_max_rad']:.3e} |"
        )
    lines.extend([
        "",
        "## 解释",
        "",
        "选定指尖对的归一化距离是功能目标；它不等于真实接触。`absolute_landmark_diagnostic` 仍记录机械代理点与 R 输入点的绝对误差，但该误差不参与 v15 优化。碰撞、接触法向、物体和执行器命令 `u` 尚未评估。",
    ])
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "actions": [item["name"] for item in results]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
