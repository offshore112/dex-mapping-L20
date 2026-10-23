"""Run a deterministic R landmark -> L20 q constrained optimization baseline."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared_r_l20 import SharedRL20  # noqa: E402


def solve_action(model: SharedRL20, action: dict, starts: list[np.ndarray]) -> dict:
    from scipy.optimize import least_squares

    target = np.asarray(action["target_landmarks_m"], dtype=float)
    weights = np.ones(target.shape[0], dtype=float)
    # Keep all 21 landmark errors in the objective; emphasize action tips only
    # for the diagnostic pair, without changing the target definition.
    if action.get("tip_pair"):
        for name in action["tip_pair"]:
            weights[model.manifest["kinematics"]["point_order"].index(name)] = 2.0
    sqrt_w = np.sqrt(weights)[:, None]
    bounds = np.asarray([model.robot.bounds[name] for name in model.q_order], dtype=float)

    def residual(q):
        return ((model.landmarks(q) - target) * sqrt_w).reshape(-1) / 0.1

    trials = []
    for start_id, x0 in enumerate(starts):
        result = least_squares(
            residual, np.clip(x0, bounds[:, 0], bounds[:, 1]),
            bounds=(bounds[:, 0], bounds[:, 1]), method="trf",
            xtol=1e-11, ftol=1e-11, gtol=1e-11, max_nfev=600,
            diff_step=1e-5,
        )
        actual = model.landmarks(result.x)
        error = actual - target
        full = model.expand_q(result.x)
        mimic_residuals = []
        for joint in model.manifest["kinematics"]["joints"]:
            if joint["mimic"] is not None:
                m = joint["mimic"]
                mimic_residuals.append(full[joint["name"]] - (m["multiplier"] * full[m["joint"]] + m["offset"]))
        trials.append({
            "start_id": start_id,
            "success": bool(result.success),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "cost": float(result.cost),
            "q": result.x,
            "full_q": full,
            "error": error,
            "mimic_residual_rad": float(max(np.abs(mimic_residuals), default=0.0)),
        })
    best = min(trials, key=lambda x: x["cost"])
    point_error_mm = np.linalg.norm(best["error"], axis=1) * 1000
    points = model.manifest["kinematics"]["point_order"]
    tip_metrics = {}
    for name in ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]:
        tip_metrics[name] = {
            "target_xyz_m": target[points.index(name)].tolist(),
            "actual_xyz_m": model.landmarks(best["q"])[points.index(name)].tolist(),
            "point_error_mm": float(point_error_mm[points.index(name)]),
        }
    if action.get("tip_pair"):
        i, j = [points.index(x) for x in action["tip_pair"]]
        target_distance = float(np.linalg.norm(target[i] - target[j]) * 1000)
        actual_points = model.landmarks(best["q"])
        actual_distance = float(np.linalg.norm(actual_points[i] - actual_points[j]) * 1000)
        pair_metrics = {
            "pair": action["tip_pair"],
            "target_distance_mm": target_distance,
            "actual_distance_mm": actual_distance,
            "distance_error_mm": actual_distance - target_distance,
            "diagnostic_threshold_20mm": actual_distance <= 20.0,
        }
    else:
        pair_metrics = None
    return {
        "name": action["name"],
        "kind": action["kind"],
        "tip_pair": action.get("tip_pair"),
        "target_tip_distance_mm": action.get("target_tip_distance_mm"),
        "solver": {
            "multi_start_count": len(starts),
            "best_start_id": best["start_id"],
            "success": best["success"],
            "message": best["message"],
            "nfev": best["nfev"],
            "cost": best["cost"],
        },
        "q_order": model.q_order,
        "q_independent_rad": best["q"].tolist(),
        "q_full_rad": best["full_q"],
        "landmark_rmse_mm": float(np.sqrt(np.mean(point_error_mm ** 2))),
        "landmark_mean_mm": float(np.mean(point_error_mm)),
        "landmark_max_mm": float(np.max(point_error_mm)),
        "tip_metrics": tip_metrics,
        "pair_metrics": pair_metrics,
        "limit_violations": 0,
        "mimic_residual_max_rad": best["mimic_residual_rad"],
    }


def solve_pair_only(model: SharedRL20, action: dict) -> dict | None:
    """Diagnostic: optimize only one tip pair and freeze other fingers at q=0."""
    from scipy.optimize import least_squares

    if not action.get("tip_pair"):
        return None
    pair = action["tip_pair"]
    points = model.manifest["kinematics"]["point_order"]
    i, j = [points.index(x) for x in pair]
    finger = "index" if "index_tip" in pair else "middle"
    active_names = [
        f"{finger}_mcp_roll", f"{finger}_mcp_pitch", f"{finger}_pip",
        "thumb_cmc_yaw", "thumb_cmc_roll", "thumb_cmc_pitch", "thumb_mcp",
    ]
    active = [model.q_order.index(name) for name in active_names]
    bounds = np.asarray([model.robot.bounds[name] for name in active_names], dtype=float)

    def unpack(x):
        q = np.zeros(len(model.q_order), dtype=float)
        q[active] = x
        return q

    def residual(x):
        q = unpack(x)
        return (model.landmarks(q)[i] - model.landmarks(q)[j]) * 10.0

    starts = [
        np.zeros(len(active_names), dtype=float),
        (bounds[:, 0] + bounds[:, 1]) / 2,
        bounds[:, 0] + 0.5 * (bounds[:, 1] - bounds[:, 0]),
    ]
    trials = []
    for start_id, start in enumerate(starts):
        result = least_squares(
            residual, np.clip(start, bounds[:, 0], bounds[:, 1]),
            bounds=(bounds[:, 0], bounds[:, 1]), max_nfev=800,
            xtol=1e-12, ftol=1e-12, gtol=1e-12, diff_step=1e-5,
        )
        q = unpack(result.x)
        actual_points = model.landmarks(q)
        distance = float(np.linalg.norm(actual_points[i] - actual_points[j]) * 1000)
        trials.append((distance, start_id, result, q))
    distance, start_id, result, q = min(trials, key=lambda item: item[0])
    full = model.expand_q(q)
    mimic_residuals = []
    for joint in model.manifest["kinematics"]["joints"]:
        if joint["mimic"] is not None:
            m = joint["mimic"]
            mimic_residuals.append(full[joint["name"]] - (m["multiplier"] * full[m["joint"]] + m["offset"]))
    return {
        "pair": pair,
        "active_q": active_names,
        "inactive_q_fixed_at_zero": [name for name in model.q_order if name not in active_names],
        "best_start_id": start_id,
        "solver_success": bool(result.success),
        "solver_message": str(result.message),
        "nfev": int(result.nfev),
        "target_distance_mm": float(action.get("target_tip_distance_mm", float("nan"))),
        "minimum_actual_distance_mm": distance,
        "distance_error_to_R_target_mm": distance - float(action.get("target_tip_distance_mm", 0.0)),
        "q_independent_rad": q.tolist(),
        "q_full_rad": full,
        "limit_violations": 0,
        "mimic_residual_max_rad": float(max(np.abs(mimic_residuals), default=0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = SharedRL20(args.model.resolve())
    actions_payload = json.loads(args.actions.resolve().read_text(encoding="utf-8"))
    bounds = np.asarray([model.robot.bounds[name] for name in model.q_order], dtype=float)
    rng = np.random.default_rng(20261017)
    starts = [
        np.zeros(len(model.q_order), dtype=float),
        (bounds[:, 0] + bounds[:, 1]) / 2,
        bounds[:, 0] + 0.25 * (bounds[:, 1] - bounds[:, 0]),
        rng.uniform(bounds[:, 0], bounds[:, 1]),
    ]
    results = [solve_action(model, action, starts) for action in actions_payload["actions"]]
    task_only = [value for value in (solve_pair_only(model, action) for action in actions_payload["actions"]) if value is not None]
    payload = {
        "version": "v14_R_L20_constraint_optimization_baseline",
        "status": "completed_synthetic_R_to_L20_q_baseline",
        "model": str(args.model.resolve()),
        "actions": str(args.actions.resolve()),
        "optimizer": {
            "method": "scipy.least_squares bounded multi-start",
            "objective": "21 R landmark positions in hand_base_link; action tip pair weight=2",
            "starts": 4,
            "q_space": "16 independent L20 coordinates; mimic expanded by URDF",
            "u_space": "not defined; no actuator descriptor",
        },
        "diagnostic_thresholds": {
            "tip_distance_20mm": "smoke diagnostic only; not frozen formal metric",
        },
        "results": results,
        "task_only_pair_diagnostics": task_only,
        "limitations": [
            "R targets are synthetic MANO pose targets, not camera estimates or a dataset.",
            "The optimization tests q-level reachability and mimic handling, not real actuator control u.",
            "No claim of generalization or final R_L20 suitability follows from these smoke actions.",
        ],
    }
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# v14：R→L20 约束优化基线",
        "",
        "本版本固定 v13 R，使用三个可复核的合成目标：平展张开、拇食指捏合、拇中指对指。优化变量是 L20 的 16 个独立关节状态 q，5 条 mimic 由 URDF 展开；没有定义执行器 u。",
        "",
        "结果详见 `metrics.json`。20 mm 指尖距离只作为本轮 smoke diagnostic，不是冻结的正式任务阈值。目标来自固定 MANO 姿态向量，不是摄像头数据。",
        "",
        "| action | landmark RMSE (mm) | max (mm) | target pair (mm) | actual pair (mm) | pair <=20 mm | mimic residual (rad) |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for r in results:
        pair = r["pair_metrics"]
        lines.append(f"| {r['name']} | {r['landmark_rmse_mm']:.3f} | {r['landmark_max_mm']:.3f} | "
                     f"{pair['target_distance_mm']:.3f} | {pair['actual_distance_mm']:.3f} | "
                     f"{pair['diagnostic_threshold_20mm']} | {r['mimic_residual_max_rad']:.3e} |" if pair else
                     f"| {r['name']} | {r['landmark_rmse_mm']:.3f} | {r['landmark_max_mm']:.3f} | — | — | — | {r['mimic_residual_max_rad']:.3e} |")
    lines.extend([
        "",
        "## L20 机构自身的任务特征对照",
        "",
        "下面的诊断只优化对应拇指/手指的 q，其他手指固定在 q=0；它回答 L20 机构本身能否把指尖靠近，不代表 R 的全关键点映射已经成功。",
        "",
        "| pair | R target distance (mm) | L20 task-only minimum (mm) | active q count | mimic residual (rad) |",
        "|---|---:|---:|---:|---:|",
    ])
    for d in task_only:
        lines.append(f"| {' + '.join(d['pair'])} | {d['target_distance_mm']:.3f} | {d['minimum_actual_distance_mm']:.3e} | {len(d['active_q'])} | {d['mimic_residual_max_rad']:.3e} |")
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "actions": [r["name"] for r in results]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
