"""Validate a fresh v23 personal-to-R repeat against its previous run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audit_pinky_distal_v23 import digest, write_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("experiment/v23_pinky_distal_twist/run2_personal_mapping_test"))
    parser.add_argument("--reference", type=Path, default=Path("experiment/v23_pinky_distal_twist/run1_preserve_points"))
    parser.add_argument("--gui-exit-code", type=int, required=True,
                        help="actual observed exit code of the 30-second GUI check")
    args = parser.parse_args()
    run, reference = args.run.resolve(), args.reference.resolve()
    audit = read(run / "metrics.json")
    mapping = read(run / "fixed_r/metrics.json")
    config = read(run / "config.json")
    prior_config = read(reference / "config.json")
    model_check = read(run / "native_model_check.json")
    scenes = {view: read(run / "mujoco" / view / "scene_check.json") for view in ("separate", "r-only")}
    renders = {view: read(run / "mujoco" / view / "render_info.json") for view in scenes}
    if config["input_sha256"] != prior_config["input_sha256"]:
        raise ValueError("Repeat uses a different original sequence")
    if digest(Path(config["input"])) != config["input_sha256"]:
        raise ValueError("Original input changed")
    array_differences = {}
    with np.load(run / "fixed_r/motion_geometry.npz", allow_pickle=False) as current, np.load(reference / "fixed_r/motion_geometry.npz", allow_pickle=False) as previous:
        if set(current.files) != set(previous.files):
            raise ValueError("Output array contracts differ")
        for key in current.files:
            if current[key].shape != previous[key].shape:
                raise ValueError(f"Array shape differs: {key}")
            array_differences[key] = float(np.max(np.abs(current[key] - previous[key])))
        count = len(current["frame_ids"])
    for view, scene in scenes.items():
        if scene["checked_frame_count"] != count or scene["skin_vertex_max_abs_error_m"] > 1e-6:
            raise ValueError(f"Full-frame skin check failed: {view}")
    if (count != audit["frame_count"] or count != mapping["frame_count"]
            or not mapping["target_beta_fixed"]
            or mapping["wrist_anchor_max_abs_error_m"] != 0
            or mapping["local_pose_transfer_max_abs_error_rad"] != 0
            or audit["unchanged_pose_components_max_abs_rad"] != 0
            or args.gui_exit_code != 0):
        raise ValueError("Mapping contract or observed GUI exit failed")
    summary = {
        "version": "v23_pinky_distal_twist", "run_id": run.name,
        "status": "passed_existing_sequence_mapping_repeat_and_display_checks",
        "input": config["input"], "input_sha256": config["input_sha256"],
        "scope": "same personal recording, fresh correction and transfer; no new capture, streaming or network training",
        "frame_count": count, "duration_s": mapping["duration_s"], "frame_drop_count": 0,
        "reference_run": str(reference), "reference_array_max_abs_differences": array_differences,
        "repeat_arrays_identical": all(value == 0 for value in array_differences.values()),
        "target_beta_fixed": mapping["target_beta_fixed"], "target_beta_R": mapping["target_beta_R"],
        "target_scale_R": mapping["target_scale_R"],
        "wrist_anchor_max_abs_error_m": mapping["wrist_anchor_max_abs_error_m"],
        "source_reconstruction_max_abs_error_m": mapping["source_reconstruction_max_abs_error_m"],
        "bone_direction_rmse_deg": mapping["bone_direction_rmse_deg"],
        "tip_pair_mean_abs_error_normalized": mapping["tip_pair_mean_abs_error_normalized"],
        "initial60_source_dip_twist_abs_mean_deg": audit["initial60_twist_abs_deg_after"]["mean"],
        "frame0_source_dip_twist_deg": audit["diagnostic_frames"][0]["twist_deg_after"],
        "raw_capture_canonicalization_max_abs_m": model_check["raw_capture_canonicalization_max_abs_m"],
        "mujoco_scene_checks": scenes, "render_outputs": renders,
        "gui_check": {"view": "separate", "requested_seconds": 30, "observed_exit_code": args.gui_exit_code,
                      "note": "bounded GUI run; full-frame vertex checks are recorded separately"},
        "script_sha256": digest(Path(__file__)),
        "viewer_sha256": digest(Path(__file__).with_name("view_fixed_r_mujoco_v22.py")),
        "limitations": ["Single sequence repeat, not an independent new-action or cross-operator test.",
                        "Neutral twist prior; no ground-truth nail orientation.",
                        "Visual kinematics only; no L20 q/u, collision/contact or live camera test."],
    }
    write_json(run / "mapping_test_summary.json", summary)
    print(json.dumps({key: summary[key] for key in ("status", "frame_count", "duration_s", "repeat_arrays_identical",
                     "bone_direction_rmse_deg", "tip_pair_mean_abs_error_normalized", "gui_check")}), flush=True)


if __name__ == "__main__":
    main()
