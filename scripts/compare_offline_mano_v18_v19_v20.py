"""Run the offline MANO -> R -> L20 interface and compare v18/v19 stability."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_command(command: list[str]) -> None:
    completed = subprocess.run(command, check=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed: {command}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mano", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-rate-hz", type=float, default=30.0)
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prepared = output / "offline_mano_sequence_v20.json"
    conditioned = output / "target_conditioned_sequence_v20.json"
    v18_dir = output / "v18_sequential"
    v19_first = output / "v19_global_first"
    v19_final = output / "v19_global_continuation"
    python = sys.executable
    scripts = Path(__file__).resolve().parent

    run_command([python, str(scripts / "prepare_offline_mano_sequence_v20.py"),
                 "--input", str(args.input.resolve()), "--mano", str(args.mano.resolve()),
                 "--model", str(args.model.resolve()), "--output", str(prepared),
                 "--frame-rate-hz", str(args.frame_rate_hz)])
    run_command([python, str(scripts / "build_target_conditioned_sequence_v18.py"),
                 "--model", str(args.model.resolve()), "--sequence", str(prepared),
                 "--output", str(conditioned)])
    run_command([python, str(scripts / "run_r_l20_alignment_v17.py"), "--version", "v20_v18",
                 "--model", str(args.model.resolve()), "--sequence", str(conditioned),
                 "--output", str(v18_dir), "--smooth-weight", "0.35"])
    run_command([python, str(scripts / "run_r_l20_global_trajectory_v19.py"),
                 "--model", str(args.model.resolve()), "--sequence", str(conditioned),
                 "--initial", str(v18_dir / "sequence_q.json"), "--output", str(v19_first),
                 "--smooth-weight", "0.80", "--max-nfev", "35"])
    first_metrics = json.loads((v19_first / "metrics.json").read_text(encoding="utf-8"))
    if first_metrics["optimizer"]["success"]:
        v19_final = v19_first
    else:
        run_command([python, str(scripts / "run_r_l20_global_trajectory_v19.py"),
                     "--model", str(args.model.resolve()), "--sequence", str(conditioned),
                     "--initial", str(v19_first / "sequence_q.json"), "--output", str(v19_final),
                     "--smooth-weight", "0.80", "--max-nfev", "35"])

    v18 = json.loads((v18_dir / "metrics.json").read_text(encoding="utf-8"))
    v19 = json.loads((v19_final / "metrics.json").read_text(encoding="utf-8"))
    s18, s19 = v18["summary"], v19["summary"]
    comparison = {
        "version": "v20_offline_MANO_v18_v19_stability_comparison",
        "status": "completed_offline_interface_fixture_comparison",
        "input": str(args.input.resolve()),
        "prepared_sequence": str(prepared),
        "target_conditioned_sequence": str(conditioned),
        "methods": {
            "v18_sequential": {"metrics": str(v18_dir / "metrics.json"), "summary": s18},
            "v19_global": {"metrics": str(v19_final / "metrics.json"), "summary": s19},
        },
        "stability_delta_v19_minus_v18": {
            "max_q_step_abs_rad": s19["max_q_step_abs_rad"] - s18["max_q_step_abs_rad"],
            "mean_q_step_abs_rad": s19["mean_q_step_abs_rad"] - s18["mean_q_step_abs_rad"],
            "max_q_step_norm_normalized": s19["max_q_step_norm_normalized"] - s18["max_q_step_norm_normalized"],
            "semantic_point_rmse_normalized": s19["mean_semantic_point_rmse_normalized"] - s18["mean_semantic_point_rmse_normalized"],
            "bone_direction_rmse_deg": s19["mean_bone_direction_rmse_deg"] - s18["mean_bone_direction_rmse_deg"],
        },
        "interpretation": "v19 is more stable if q-step deltas are negative; this fixture is an offline synthetic MANO replay, not a camera generalization result.",
        "limitations": [
            "The supplied fixture is generated MANO, not a real camera recording.",
            "The comparison uses the same fixed target-conditioned adapter and L20 model for both methods.",
            "No collision/contact or actuator u is included.",
        ],
    }
    (output / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# v20：离线 MANO 序列接口与 v18/v19 稳定性比较",
        "",
        "本目录把输入统一为离线 MANO 序列，再通过同一个 L20 目标条件化适配器比较 v18 逐帧求解和 v19 整段联合优化。当前回放使用项目已有的固定合成序列作为接口 fixture，不是真实摄像头数据。",
        "",
        "## 结论",
        "",
        f"- v18 最大/平均 q 单步：`{s18['max_q_step_abs_rad']:.6f}/{s18['mean_q_step_abs_rad']:.6f} rad`。",
        f"- v19 最大/平均 q 单步：`{s19['max_q_step_abs_rad']:.6f}/{s19['mean_q_step_abs_rad']:.6f} rad`。",
        f"- v19 相对 v18 的最大 q 单步变化差：`{comparison['stability_delta_v19_minus_v18']['max_q_step_abs_rad']:.6f} rad`。",
        f"- v18/v19 原始任务对平均归一化误差：`{s18['pair_mean_abs_normalized_error']}` / `{s19['pair_mean_abs_normalized_error']}`。",
        "",
        "输入格式说明见 `offline_mano_sequence_v20.json` 的 metadata 和 `scripts/prepare_offline_mano_sequence_v20.py`。",
    ]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": comparison["status"], "output": str(output),
                      "v18_max_q_step": s18["max_q_step_abs_rad"],
                      "v19_max_q_step": s19["max_q_step_abs_rad"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
