"""Run v17 functional-skeleton alignment on the v16 synthetic sequence.

Compared with v15/v16, the objective explicitly aligns all semantic points in
each hand's own palm-normalized frame and the bend angle between consecutive
bone segments. Absolute MANO/L20 positions remain a diagnostic only. The
selected pinch/opposition pair keeps task priority, and adjacent q values use
the v16 temporal smoothness term.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_r_l20_feature_baseline_v15 import (  # noqa: E402
    CHAINS,
    POINT_ORDER,
    make_feature_record,
    mimic_residual,
    selected_pair_index,
)
from shared_r_l20 import SharedRL20  # noqa: E402


def alignment_arrays(record: dict) -> dict:
    points = np.asarray(record["canonical_points"], dtype=float)
    edges = record["edges"]
    directions = np.asarray([edge["direction"] for edge in edges], dtype=float)
    lengths = np.asarray([edge["length_normalized"] for edge in edges], dtype=float)
    bend_angles = []
    edge_index = 0
    for chain in CHAINS.values():
        edge_count = len(chain) - 1
        chain_dirs = directions[edge_index:edge_index + edge_count]
        for first, second in zip(chain_dirs[:-1], chain_dirs[1:]):
            bend_angles.append(np.arccos(np.clip(float(np.dot(first, second)), -1.0, 1.0)))
        edge_index += edge_count
    tip_pairs = record["tip_pairs"]
    return {
        "points": points[1:],  # all non-wrist semantic points in the palm frame
        "directions": directions,
        "lengths": lengths,
        "bend_angles": np.asarray(bend_angles, dtype=float),
        "tip_distances": np.asarray([item["distance_normalized"] for item in tip_pairs], dtype=float),
        "tip_directions": np.asarray([item["direction"] for item in tip_pairs], dtype=float),
    }


def feature_weights(action: dict) -> dict:
    weights = {
        "direction": 1.25,
        "length": 0.45,
        "points": 1.10,
        "bend": 0.85,
        "pair_distance": 0.35,
        "pair_direction": 0.35,
        "selected_pair_distance": 2.5,
        "selected_pair_direction": 0.8,
        "pair_direction_min_distance": 0.025,
        "selected_pair": action.get("tip_pair"),
    }
    if action.get("tip_pair"):
        # Preserve the functional task first, while still aligning the full
        # semantic skeleton as a lower-weight regularizer.
        weights.update({
            "direction": 0.75,
            "length": 0.25,
            "points": 0.35,
            "bend": 0.30,
            "pair_distance": 0.15,
            "pair_direction": 0.15,
            "selected_pair_distance": 35.0,
            "selected_pair_direction": 0.35,
        })
    return weights


def residual_for(q: np.ndarray, model: SharedRL20, target_record: dict, weights: dict,
                 previous: np.ndarray | None, q_range: np.ndarray, smooth_weight: float,
                 task_record: dict | None = None) -> np.ndarray:
    actual_record = make_feature_record(model.landmarks(q))
    actual = alignment_arrays(actual_record)
    target = alignment_arrays(target_record)
    task = target if task_record is None else alignment_arrays(task_record)
    pieces = [
        (actual["directions"] - target["directions"]).reshape(-1) * weights["direction"],
        (actual["lengths"] - target["lengths"]).reshape(-1) * weights["length"],
        (actual["points"] - target["points"]).reshape(-1) * weights["points"],
        (actual["bend_angles"] - target["bend_angles"]).reshape(-1) * weights["bend"],
        (actual["tip_distances"] - target["tip_distances"]).reshape(-1) * weights["pair_distance"],
        (actual["tip_directions"] - target["tip_directions"]).reshape(-1) * weights["pair_direction"],
    ]
    pair_i = selected_pair_index(weights.get("selected_pair"))
    if pair_i is not None:
        pieces.append(np.asarray([
            (actual["tip_distances"][pair_i] - task["tip_distances"][pair_i])
            * weights["selected_pair_distance"]
        ]))
        if task["tip_distances"][pair_i] >= weights["pair_direction_min_distance"]:
            pieces.append((actual["tip_directions"][pair_i] - task["tip_directions"][pair_i])
                          * weights["selected_pair_direction"])
    if previous is not None:
        pieces.append((q - previous) / q_range * smooth_weight)
    return np.concatenate(pieces)


def solve_frame(model: SharedRL20, action: dict, previous: np.ndarray | None,
                bounds: np.ndarray, rng: np.random.Generator, smooth_weight: float) -> dict:
    from scipy.optimize import least_squares

    target_points = np.asarray(action["target_landmarks_m"], dtype=float)
    target_record = make_feature_record(target_points)
    task_points = np.asarray(action.get("task_target_landmarks_m", action["target_landmarks_m"]), dtype=float)
    task_record = make_feature_record(task_points)
    weights = feature_weights(action)
    q_range = np.maximum(bounds[:, 1] - bounds[:, 0], 1e-9)
    midpoint = (bounds[:, 0] + bounds[:, 1]) / 2.0
    starts = [
        previous.copy() if previous is not None else np.zeros(len(model.q_order), dtype=float),
        midpoint,
        bounds[:, 0] + 0.25 * (bounds[:, 1] - bounds[:, 0]),
        rng.uniform(bounds[:, 0], bounds[:, 1]),
    ]
    trials = []
    for start_id, start in enumerate(starts):
        result = least_squares(
            lambda q: residual_for(q, model, target_record, weights, previous, q_range, smooth_weight, task_record),
            np.clip(start, bounds[:, 0], bounds[:, 1]),
            bounds=(bounds[:, 0], bounds[:, 1]), method="trf",
            xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=900, diff_step=1e-5,
        )
        trials.append((float(result.cost), start_id, result))
    cost, start_id, result = min(trials, key=lambda item: item[0])
    q = result.x
    actual_points = model.landmarks(q)
    actual_record = make_feature_record(actual_points)
    actual = alignment_arrays(actual_record)
    target = alignment_arrays(target_record)
    task = alignment_arrays(task_record)
    direction_errors = np.degrees(np.arccos(np.clip(np.sum(actual["directions"] * target["directions"], axis=1), -1.0, 1.0)))
    bend_errors = np.degrees(np.abs(actual["bend_angles"] - target["bend_angles"]))
    tip_direction_errors = np.degrees(np.arccos(np.clip(np.sum(actual["tip_directions"] * target["tip_directions"], axis=1), -1.0, 1.0)))
    pair_i = selected_pair_index(action.get("tip_pair"))
    pair_metrics = None
    if pair_i is not None:
        pair_metrics = {
            "pair": action["tip_pair"],
            "target_distance_normalized": float(task["tip_distances"][pair_i]),
            "actual_distance_normalized": float(actual["tip_distances"][pair_i]),
            "normalized_distance_error": float(actual["tip_distances"][pair_i] - task["tip_distances"][pair_i]),
            "target_distance_mm_in_R_input": float(task["tip_distances"][pair_i] * task_record["palm_scale_m"] * 1000.0),
            "actual_distance_mm_in_L20_root": float(actual["tip_distances"][pair_i] * actual_record["palm_scale_m"] * 1000.0),
            "direction_error_deg": float(tip_direction_errors[pair_i]),
            "direction_used_in_objective": bool(task["tip_distances"][pair_i] >= weights["pair_direction_min_distance"]),
            "target_source": "original_R_task_target" if "task_target_landmarks_m" in action else "alignment_target",
        }
    absolute = actual_points - target_points
    q_step = None if previous is None else q - previous
    return {
        "frame_id": int(action["frame_id"]),
        "name": action["name"],
        "phase": action["phase"],
        "alpha": float(action["alpha"]),
        "tip_pair": action.get("tip_pair"),
        "task_target_source": "original_R_task_target" if "task_target_landmarks_m" in action else "alignment_target",
        "solver": {
            "success": bool(result.success), "message": str(result.message),
            "nfev": int(result.nfev), "best_start_id": int(start_id),
            "multi_start_count": len(trials), "cost": cost,
            "smooth_weight": smooth_weight, "feature_weights": weights,
        },
        "q_order": model.q_order,
        "q_independent_rad": q.tolist(),
        "q_full_rad": model.expand_q(q),
        "alignment_metrics": {
            "semantic_point_rmse_normalized": float(np.sqrt(np.mean((actual["points"] - target["points"]) ** 2))),
            "bone_direction_rmse_deg": float(np.sqrt(np.mean(direction_errors ** 2))),
            "bone_direction_max_deg": float(np.max(direction_errors)),
            "bend_angle_rmse_deg": float(np.sqrt(np.mean(bend_errors ** 2))),
            "normalized_bone_length_rmse": float(np.sqrt(np.mean((actual["lengths"] - target["lengths"]) ** 2))),
            "normalized_all_tip_pair_distance_rmse": float(np.sqrt(np.mean((actual["tip_distances"] - target["tip_distances"]) ** 2))),
            "tip_pair_direction_rmse_deg": float(np.sqrt(np.mean(tip_direction_errors ** 2))),
        },
        "selected_pair_metrics": pair_metrics,
        "absolute_landmark_diagnostic": {
            "rmse_mm": float(np.sqrt(np.mean(absolute ** 2)) * 1000.0),
            "max_mm": float(np.max(np.linalg.norm(absolute, axis=1)) * 1000.0),
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
    parser.add_argument("--smooth-weight", type=float, default=0.35)
    parser.add_argument("--version", default="v17", help="artifact version label")
    args = parser.parse_args()
    model = SharedRL20(args.model.resolve())
    if model.manifest["kinematics"]["point_order"] != POINT_ORDER:
        raise ValueError("Model point order differs from frozen v17 semantic order")
    sequence = json.loads(args.sequence.resolve().read_text(encoding="utf-8"))["sequence"]
    bounds = np.asarray([model.robot.bounds[name] for name in model.q_order], dtype=float)
    rng = np.random.default_rng(20261017)
    previous = None
    results = []
    for action in sequence:
        result = solve_frame(model, action, previous, bounds, rng, args.smooth_weight)
        results.append(result)
        previous = np.asarray(result["q_independent_rad"], dtype=float)
    pair_errors = {}
    for result in results:
        pair = result["selected_pair_metrics"]
        if pair is not None:
            pair_errors.setdefault("+".join(pair["pair"]), []).append(abs(pair["normalized_distance_error"]))
    q_steps = [r["q_step_max_abs_rad"] for r in results if r["q_step_max_abs_rad"] is not None]
    q_norms = [r["q_step_norm_normalized"] for r in results if r["q_step_norm_normalized"] is not None]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": f"{args.version}_R_L20_functional_skeleton_alignment",
        "status": "completed_synthetic_functional_skeleton_alignment",
        "model": str(args.model.resolve()), "sequence": str(args.sequence.resolve()),
        "frame_count": len(results),
        "representation": {
            "surface": "natural MANO-R surface kept separate from L20 mesh",
            "functional_skeleton": "all non-wrist semantic points in independent palm-normalized frames plus bone bend angles",
            "absolute_point_matching": False,
        },
        "optimizer": {
            "method": "sequential bounded scipy.least_squares multi-start",
            "objective": "semantic point alignment + bone directions + bend angles + task pair + q smoothness",
            "smooth_weight": args.smooth_weight,
            "q_space": "16 independent L20 q; 5 mimic joints expanded by URDF",
            "u_space": "not defined",
        },
        "summary": {
            "all_solver_success": bool(all(item["solver"]["success"] for item in results)),
            "all_limit_violations_zero": bool(all(item["limit_violations"] == 0 for item in results)),
            "max_mimic_residual_rad": float(max(item["mimic_residual_max_rad"] for item in results)),
            "mean_semantic_point_rmse_normalized": float(np.mean([item["alignment_metrics"]["semantic_point_rmse_normalized"] for item in results])),
            "mean_bone_direction_rmse_deg": float(np.mean([item["alignment_metrics"]["bone_direction_rmse_deg"] for item in results])),
            "mean_bend_angle_rmse_deg": float(np.mean([item["alignment_metrics"]["bend_angle_rmse_deg"] for item in results])),
            "max_q_step_abs_rad": float(max(q_steps, default=0.0)),
            "mean_q_step_abs_rad": float(np.mean(q_steps)) if q_steps else 0.0,
            "max_q_step_norm_normalized": float(max(q_norms, default=0.0)),
            "pair_mean_abs_normalized_error": {key: float(np.mean(values)) for key, values in pair_errors.items()},
            "pair_max_abs_normalized_error": {key: float(np.max(values)) for key, values in pair_errors.items()},
        },
        "results": results,
        "limitations": [
            "The sequence is synthetic interpolation of three fixed MANO poses, not camera data.",
            "This aligns a functional skeleton, not every human and CAD surface vertex.",
            "Collision/contact and actuator u are not active; the current collision meshes have unfiltered overlaps.",
        ],
    }
    (output / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "sequence_q.json").write_text(json.dumps({
        "version": payload["version"], "q_order": model.q_order,
        "frames": [{"frame_id": item["frame_id"], "name": item["name"],
                    "q_independent_rad": item["q_independent_rad"], "q_full_rad": item["q_full_rad"]}
                   for item in results],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / f"r_l20_{args.version}_alignment_model.json").write_text(json.dumps({
        "version": f"{args.version}_R_L20_functional_skeleton_alignment_model",
        "point_order": POINT_ORDER,
        "palm_frame": "z=wrist->middle_mcp; x=pinky_mcp->index_mcp projected; y=cross(z,x)",
        "features": ["all_non_wrist_canonical_points", "bone_direction", "bone_length_normalized", "bone_bend_angle", "tip_pair_distance"],
        "surface_source_model": str(args.model.resolve()),
        "q_order": model.q_order, "mimic_count": 5, "u_space": "not_defined",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"# {args.version}：R 与 L20 功能骨架对齐",
        "",
        f"本版本在 v16 的功能特征基础上加入所有非腕部语义点的掌部归一化位置和相邻骨段屈曲角，仍保留捏合/对指任务优先级。自然 MANO 表面不被机械网格替换；优化变量是 L20 的 16 个独立 q，smooth_weight={args.smooth_weight}。",
        "",
        "## 汇总",
        "",
        f"- 所有帧求解成功：`{payload['summary']['all_solver_success']}`；限位违规：`{not payload['summary']['all_limit_violations_zero']}`；最大 mimic 残差：`{payload['summary']['max_mimic_residual_rad']:.3e} rad`。",
        f"- 平均语义骨架点 RMSE（掌部归一化）：`{payload['summary']['mean_semantic_point_rmse_normalized']:.4f}`。",
        f"- 平均骨方向 RMSE：`{payload['summary']['mean_bone_direction_rmse_deg']:.3f}°`；平均屈曲角 RMSE：`{payload['summary']['mean_bend_angle_rmse_deg']:.3f}°`。",
        f"- 最大独立 q 单步变化：`{payload['summary']['max_q_step_abs_rad']:.6f} rad`。",
        "",
        "详细逐帧结果在 `metrics.json`，q 序列在 `sequence_q.json`。叠加图和分开图由独立渲染脚本生成。",
    ]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "frame_count": len(results), "summary": payload["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
