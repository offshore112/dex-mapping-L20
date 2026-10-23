"""Solve a continuous v16 MANO-R sequence with L20 feature and smoothness terms."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_r_l20_feature_baseline_v15 import (  # noqa: E402
    POINT_ORDER,
    feature_arrays,
    make_feature_record,
    mimic_residual,
    residual_for,
    selected_pair_index,
)
from shared_r_l20 import SharedRL20  # noqa: E402


def feature_weights(action: dict) -> dict:
    weights = {
        "direction": 1.0,
        "length": 0.65,
        "mcp": 0.75,
        "pair_distance": 0.70,
        "pair_direction": 0.45,
        "selected_pair_distance": 2.5,
        "selected_pair_direction": 1.25,
        "pair_direction_min_distance": 0.025,
        "selected_pair": action.get("tip_pair"),
    }
    if action.get("tip_pair"):
        weights.update({
            "direction": 0.50,
            "length": 0.20,
            "mcp": 0.20,
            "pair_distance": 0.15,
            "pair_direction": 0.10,
            "selected_pair_distance": 30.0,
            "selected_pair_direction": 0.20,
        })
    return weights


def starts_for(bounds: np.ndarray, previous: np.ndarray | None, rng: np.random.Generator) -> list[np.ndarray]:
    midpoint = (bounds[:, 0] + bounds[:, 1]) / 2.0
    starts = [
        np.zeros(bounds.shape[0], dtype=float),
        midpoint,
        bounds[:, 0] + 0.25 * (bounds[:, 1] - bounds[:, 0]),
        rng.uniform(bounds[:, 0], bounds[:, 1]),
    ]
    if previous is not None:
        starts[0] = previous.copy()
    return starts


def solve_frame(model: SharedRL20, action: dict, previous: np.ndarray | None,
                bounds: np.ndarray, rng: np.random.Generator, smooth_weight: float) -> dict:
    from scipy.optimize import least_squares

    target_points = np.asarray(action["target_landmarks_m"], dtype=float)
    target_record = make_feature_record(target_points)
    weights = feature_weights(action)
    q_range = np.maximum(bounds[:, 1] - bounds[:, 0], 1e-9)

    def residual(q: np.ndarray) -> np.ndarray:
        pieces = [residual_for(q, model, target_record, weights)]
        if previous is not None:
            pieces.append((q - previous) / q_range * smooth_weight)
        return np.concatenate(pieces)

    trials = []
    for start_id, start in enumerate(starts_for(bounds, previous, rng)):
        result = least_squares(
            residual, np.clip(start, bounds[:, 0], bounds[:, 1]),
            bounds=(bounds[:, 0], bounds[:, 1]), method="trf",
            xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=700,
            diff_step=1e-5,
        )
        trials.append((float(result.cost), start_id, result))
    cost, start_id, result = min(trials, key=lambda item: item[0])
    q = result.x
    actual_points = model.landmarks(q)
    actual_record = make_feature_record(actual_points)
    actual = feature_arrays(actual_record)
    target = feature_arrays(target_record)
    bone_angles = np.degrees(np.arccos(np.clip(np.sum(actual["directions"] * target["directions"], axis=1), -1.0, 1.0)))
    tip_angles = np.degrees(np.arccos(np.clip(np.sum(actual["tip_directions"] * target["tip_directions"], axis=1), -1.0, 1.0)))
    pair_metrics = None
    pair_i = selected_pair_index(action.get("tip_pair"))
    if pair_i is not None:
        pair_metrics = {
            "pair": action["tip_pair"],
            "target_distance_normalized": float(target["tip_distances"][pair_i]),
            "actual_distance_normalized": float(actual["tip_distances"][pair_i]),
            "normalized_distance_error": float(actual["tip_distances"][pair_i] - target["tip_distances"][pair_i]),
            "target_distance_mm_in_R_input": float(target["tip_distances"][pair_i] * target_record["palm_scale_m"] * 1000.0),
            "actual_distance_mm_in_L20_root": float(actual["tip_distances"][pair_i] * actual_record["palm_scale_m"] * 1000.0),
            "direction_error_deg": float(tip_angles[pair_i]),
            "direction_used_in_objective": bool(target["tip_distances"][pair_i] >= weights["pair_direction_min_distance"]),
        }
    q_step = None if previous is None else q - previous
    absolute_error = actual_points - target_points
    return {
        "frame_id": int(action["frame_id"]),
        "name": action["name"],
        "phase": action["phase"],
        "alpha": float(action["alpha"]),
        "tip_pair": action.get("tip_pair"),
        "solver": {
            "success": bool(result.success),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "best_start_id": int(start_id),
            "multi_start_count": len(trials),
            "cost": cost,
            "smooth_weight": smooth_weight,
            "feature_weights": weights,
        },
        "q_order": model.q_order,
        "q_independent_rad": q.tolist(),
        "q_full_rad": model.expand_q(q),
        "feature_metrics": {
            "bone_direction_rmse_deg": float(np.sqrt(np.mean(bone_angles ** 2))),
            "bone_direction_max_deg": float(np.max(bone_angles)),
            "normalized_bone_length_rmse": float(np.sqrt(np.mean((actual["lengths"] - target["lengths"]) ** 2))),
            "normalized_mcp_position_rmse": float(np.sqrt(np.mean((actual["mcp_points"] - target["mcp_points"]) ** 2))),
            "normalized_all_tip_pair_distance_rmse": float(np.sqrt(np.mean((actual["tip_distances"] - target["tip_distances"]) ** 2))),
            "tip_pair_direction_rmse_deg": float(np.sqrt(np.mean(tip_angles ** 2))),
        },
        "selected_pair_metrics": pair_metrics,
        "absolute_landmark_diagnostic": {
            "rmse_mm": float(np.sqrt(np.mean(absolute_error ** 2)) * 1000.0),
            "max_mm": float(np.max(np.linalg.norm(absolute_error, axis=1)) * 1000.0),
        },
        "q_step_rad": None if q_step is None else q_step.tolist(),
        "q_step_max_abs_rad": None if q_step is None else float(np.max(np.abs(q_step))),
        "q_step_norm_normalized": None if q_step is None else float(np.linalg.norm(q_step / q_range)),
        "limit_violations": 0,
        "mimic_residual_max_rad": mimic_residual(model, q),
        "collision_contact": "not_evaluated",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smooth-weight", type=float, default=0.50)
    args = parser.parse_args()
    if args.smooth_weight < 0:
        raise ValueError("smooth-weight must be non-negative")

    model = SharedRL20(args.model.resolve())
    if model.manifest["kinematics"]["point_order"] != POINT_ORDER:
        raise ValueError("Model point order differs from frozen v15 semantic order")
    sequence_payload = json.loads(args.sequence.resolve().read_text(encoding="utf-8"))
    frames = sequence_payload["sequence"]
    bounds = np.asarray([model.robot.bounds[name] for name in model.q_order], dtype=float)
    rng = np.random.default_rng(20261017)
    previous = None
    results = []
    for frame in frames:
        result = solve_frame(model, frame, previous, bounds, rng, args.smooth_weight)
        results.append(result)
        previous = np.asarray(result["q_independent_rad"], dtype=float)

    pair_errors = {}
    for result in results:
        pair = result["selected_pair_metrics"]
        if pair is not None:
            key = "+".join(pair["pair"])
            pair_errors.setdefault(key, []).append(abs(pair["normalized_distance_error"]))
    q_steps = [r["q_step_max_abs_rad"] for r in results if r["q_step_max_abs_rad"] is not None]
    q_norms = [r["q_step_norm_normalized"] for r in results if r["q_step_norm_normalized"] is not None]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "v16_R_L20_feature_sequence_smooth_baseline",
        "status": "completed_synthetic_sequence_feature_q_baseline",
        "model": str(args.model.resolve()),
        "sequence": str(args.sequence.resolve()),
        "frame_count": len(results),
        "optimizer": {
            "method": "sequential bounded scipy.least_squares multi-start",
            "objective": "v15 functional features + normalized independent-q temporal smoothness",
            "smooth_weight": args.smooth_weight,
            "q_space": "16 independent L20 q; 5 mimic joints expanded by URDF",
            "u_space": "not defined",
        },
        "summary": {
            "all_solver_success": bool(all(item["solver"]["success"] for item in results)),
            "all_limit_violations_zero": bool(all(item["limit_violations"] == 0 for item in results)),
            "max_mimic_residual_rad": float(max(item["mimic_residual_max_rad"] for item in results)),
            "mean_bone_direction_rmse_deg": float(np.mean([item["feature_metrics"]["bone_direction_rmse_deg"] for item in results])),
            "max_q_step_abs_rad": float(max(q_steps, default=0.0)),
            "mean_q_step_abs_rad": float(np.mean(q_steps)) if q_steps else 0.0,
            "max_q_step_norm_normalized": float(max(q_norms, default=0.0)),
            "pair_mean_abs_normalized_error": {key: float(np.mean(values)) for key, values in pair_errors.items()},
            "pair_max_abs_normalized_error": {key: float(np.max(values)) for key, values in pair_errors.items()},
        },
        "results": results,
        "limitations": [
            "The sequence is a deterministic interpolation of three synthetic MANO poses, not camera data.",
            "Collision and contact terms are not active because the current URDF collision meshes have unfiltered overlaps.",
            "The smoothness term is a q-level baseline and does not model actuator dynamics or control latency.",
            "Near-zero fingertip distance is not called contact or grasp success.",
        ],
    }
    (output / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "sequence_q.json").write_text(json.dumps({
        "version": payload["version"],
        "q_order": model.q_order,
        "frames": [{"frame_id": item["frame_id"], "name": item["name"],
                    "q_independent_rad": item["q_independent_rad"], "q_full_rad": item["q_full_rad"]}
                   for item in results],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# v16：连续 R→L20 功能特征与时间平滑基线",
        "",
        f"本版本将 v15 的功能特征 q 求解扩展到 {len(results)} 帧合成序列，并在相邻帧的 16 个独立 q 上加入归一化平滑项，smooth_weight={args.smooth_weight}。序列由张开→拇食指捏合→拇中指对指三个固定 MANO 姿态线性插值得到。",
        "",
        "## 汇总",
        "",
        f"- 所有帧求解成功：`{payload['summary']['all_solver_success']}`；限位违规：`{not payload['summary']['all_limit_violations_zero']}`；最大 mimic 残差：`{payload['summary']['max_mimic_residual_rad']:.3e} rad`。",
        f"- 平均骨方向 RMSE：`{payload['summary']['mean_bone_direction_rmse_deg']:.3f}°`。",
        f"- 相邻帧最大独立 q 变化：`{payload['summary']['max_q_step_abs_rad']:.6f} rad`；归一化 q 步长最大值：`{payload['summary']['max_q_step_norm_normalized']:.6f}`。",
        "- 碰撞/接触项未启用；当前 URDF 碰撞网格的重叠诊断见 `failure_cases.md`。",
        "",
        "详细每帧指标在 `metrics.json`，实际 q 序列在 `sequence_q.json`。",
    ]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "frame_count": len(results), "summary": payload["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
