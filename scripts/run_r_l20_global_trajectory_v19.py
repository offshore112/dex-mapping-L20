"""Jointly optimize a v18 target-conditioned q trajectory with smoothness."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_r_l20_alignment_v17 import alignment_arrays, feature_weights, residual_for  # noqa: E402
from run_r_l20_feature_baseline_v15 import make_feature_record, mimic_residual, selected_pair_index  # noqa: E402
from shared_r_l20 import SharedRL20  # noqa: E402


def evaluate_frame(model: SharedRL20, action: dict, q: np.ndarray, previous: np.ndarray | None,
                   q_range: np.ndarray, smooth_weight: float, global_solver: dict) -> dict:
    target_points = np.asarray(action["target_landmarks_m"], dtype=float)
    task_points = np.asarray(action.get("task_target_landmarks_m", action["target_landmarks_m"]), dtype=float)
    target_record = make_feature_record(target_points)
    task_record = make_feature_record(task_points)
    actual_points = model.landmarks(q)
    actual_record = make_feature_record(actual_points)
    target = alignment_arrays(target_record)
    task = alignment_arrays(task_record)
    actual = alignment_arrays(actual_record)
    direction_errors = np.degrees(np.arccos(np.clip(np.sum(actual["directions"] * target["directions"], axis=1), -1.0, 1.0)))
    bend_errors = np.degrees(np.abs(actual["bend_angles"] - target["bend_angles"]))
    tip_direction_errors = np.degrees(np.arccos(np.clip(np.sum(actual["tip_directions"] * task["tip_directions"], axis=1), -1.0, 1.0)))
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
            "target_source": "original_R_task_target" if "task_target_landmarks_m" in action else "alignment_target",
        }
    absolute = actual_points - target_points
    q_step = None if previous is None else q - previous
    return {
        "frame_id": int(action["frame_id"]), "name": action["name"], "phase": action["phase"],
        "alpha": float(action["alpha"]), "tip_pair": action.get("tip_pair"),
        "task_target_source": "original_R_task_target" if "task_target_landmarks_m" in action else "alignment_target",
        "q_order": model.q_order, "q_independent_rad": q.tolist(), "q_full_rad": model.expand_q(q),
        "solver": global_solver,
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
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smooth-weight", type=float, default=0.80)
    parser.add_argument("--max-nfev", type=int, default=45)
    args = parser.parse_args()

    from scipy.optimize import least_squares

    model = SharedRL20(args.model.resolve())
    sequence = json.loads(args.sequence.resolve().read_text(encoding="utf-8"))["sequence"]
    initial = json.loads(args.initial.resolve().read_text(encoding="utf-8"))["frames"]
    if len(sequence) != len(initial):
        raise ValueError("Sequence and initial q frame counts differ")
    q_count = len(model.q_order)
    bounds_one = np.asarray([model.robot.bounds[name] for name in model.q_order], dtype=float)
    q_range = np.maximum(bounds_one[:, 1] - bounds_one[:, 0], 1e-9)
    x0 = np.asarray([frame["q_independent_rad"] for frame in initial], dtype=float)
    if x0.shape != (len(sequence), q_count):
        raise ValueError(f"Expected initial q shape {(len(sequence), q_count)}, got {x0.shape}")
    lower = np.tile(bounds_one[:, 0], len(sequence))
    upper = np.tile(bounds_one[:, 1], len(sequence))
    weights = [feature_weights(action) for action in sequence]
    target_records = [make_feature_record(np.asarray(action["target_landmarks_m"], dtype=float)) for action in sequence]
    task_records = [make_feature_record(np.asarray(action.get("task_target_landmarks_m", action["target_landmarks_m"]), dtype=float)) for action in sequence]

    def residual_flat(flat: np.ndarray) -> np.ndarray:
        q_all = flat.reshape(len(sequence), q_count)
        pieces = []
        for index, q in enumerate(q_all):
            pieces.append(residual_for(q, model, target_records[index], weights[index], None, q_range, 0.0, task_records[index]))
            if index > 0:
                pieces.append((q - q_all[index - 1]) / q_range * args.smooth_weight)
        return np.concatenate(pieces)

    result = least_squares(
        residual_flat, np.clip(x0.reshape(-1), lower, upper), bounds=(lower, upper),
        method="trf", x_scale="jac", xtol=1e-9, ftol=1e-9, gtol=1e-9,
        max_nfev=args.max_nfev, diff_step=1e-5, verbose=0,
    )
    q_all = result.x.reshape(len(sequence), q_count)
    solver_info = {
        "method": "joint bounded scipy.least_squares over all frames",
        "success": bool(result.success), "message": str(result.message),
        "nfev": int(result.nfev), "cost": float(result.cost),
        "smooth_weight": args.smooth_weight, "max_nfev": args.max_nfev,
        "q_space": "16 independent L20 q; 5 mimic joints expanded by URDF",
        "u_space": "not defined",
    }
    frames = []
    previous = None
    for action, q in zip(sequence, q_all):
        frames.append(evaluate_frame(model, action, q, previous, q_range, args.smooth_weight, solver_info))
        previous = q
    q_steps = [item["q_step_max_abs_rad"] for item in frames if item["q_step_max_abs_rad"] is not None]
    q_norms = [item["q_step_norm_normalized"] for item in frames if item["q_step_norm_normalized"] is not None]
    pair_errors = {}
    for item in frames:
        pair = item["selected_pair_metrics"]
        if pair is not None:
            pair_errors.setdefault("+".join(pair["pair"]), []).append(abs(pair["normalized_distance_error"]))
    summary = {
        "all_solver_success": bool(result.success),
        "all_limit_violations_zero": bool(all(item["limit_violations"] == 0 for item in frames)),
        "max_mimic_residual_rad": float(max(item["mimic_residual_max_rad"] for item in frames)),
        "mean_semantic_point_rmse_normalized": float(np.mean([item["alignment_metrics"]["semantic_point_rmse_normalized"] for item in frames])),
        "mean_bone_direction_rmse_deg": float(np.mean([item["alignment_metrics"]["bone_direction_rmse_deg"] for item in frames])),
        "mean_bend_angle_rmse_deg": float(np.mean([item["alignment_metrics"]["bend_angle_rmse_deg"] for item in frames])),
        "max_q_step_abs_rad": float(max(q_steps, default=0.0)),
        "mean_q_step_abs_rad": float(np.mean(q_steps)) if q_steps else 0.0,
        "max_q_step_norm_normalized": float(max(q_norms, default=0.0)),
        "pair_mean_abs_normalized_error": {key: float(np.mean(values)) for key, values in pair_errors.items()},
        "pair_max_abs_normalized_error": {key: float(np.max(values)) for key, values in pair_errors.items()},
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "v19_R_L20_global_trajectory_target_conditioned",
        "status": "completed_joint_synthetic_trajectory_optimization",
        "model": str(args.model.resolve()), "sequence": str(args.sequence.resolve()),
        "initial": str(args.initial.resolve()), "frame_count": len(frames),
        "optimizer": solver_info, "summary": summary, "results": frames,
        "limitations": [
            "The sequence is synthetic and the target-conditioned shape target changes relative to raw MANO points.",
            "Joint optimization does not add collision/contact or actuator u.",
            "The global solver is an offline baseline, not an online controller.",
        ],
    }
    (output / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "sequence_q.json").write_text(json.dumps({
        "version": payload["version"], "q_order": model.q_order,
        "frames": [{"frame_id": item["frame_id"], "name": item["name"],
                    "q_independent_rad": item["q_independent_rad"], "q_full_rad": item["q_full_rad"]}
                   for item in frames],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# v19：目标条件化骨架的整段轨迹联合优化\n\n"
        f"本版本把 v18 的 {len(frames)} 帧 q 一起优化，而不是逐帧贪心求解；目标同时包含 L20 条件化骨架、原始 R 任务指尖对和相邻 q 平滑项，smooth_weight={args.smooth_weight}。\n\n"
        f"- 求解成功：`{summary['all_solver_success']}`；限位违规：`{not summary['all_limit_violations_zero']}`；mimic 残差：`{summary['max_mimic_residual_rad']:.3e} rad`。\n"
        f"- 语义点 RMSE（条件化目标）：`{summary['mean_semantic_point_rmse_normalized']:.6f}`；骨方向 RMSE：`{summary['mean_bone_direction_rmse_deg']:.3f}°`；屈曲角 RMSE：`{summary['mean_bend_angle_rmse_deg']:.3f}°`。\n"
        f"- 最大 q 单步变化：`{summary['max_q_step_abs_rad']:.6f} rad`；平均 q 单步变化：`{summary['mean_q_step_abs_rad']:.6f} rad`。\n"
        f"- 原始 R 任务对平均归一化误差：`{summary['pair_mean_abs_normalized_error']}`。\n\n"
        "这是离线联合优化基线，尚未加入碰撞、接触或执行器 u。\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
