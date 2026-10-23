"""Recheck MANO semantics, capture coordinates and temporal changes for v23."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from audit_pinky_distal_v23 import digest, summary, write_json
from check_mano_model import CHAINS, TIP_VERTEX_IDS, canonical_landmarks, read_mano
from fit_mediapipe_to_mano_v21 import batched_landmarks, load_capture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("experiment/v23_pinky_distal_twist/run1_preserve_points"))
    args = parser.parse_args()
    run = args.run.resolve()
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    source = json.loads(Path(config["input"]).read_text(encoding="utf-8"))
    corrected = json.loads((run / "mano_sequence_v23.json").read_text(encoding="utf-8"))
    if digest(Path(config["input"])) != config["input_sha256"]:
        raise ValueError("Original v21 sequence changed")
    if source["shape_fit"] != corrected["shape_fit"]:
        raise ValueError("Source morphology changed")
    before = np.asarray([f["theta_hand_pose"] for f in source["frames"]])
    after = np.asarray([f["theta_hand_pose"] for f in corrected["frames"]])
    timestamps = np.asarray([f["timestamp_s"] for f in source["frames"]])
    if not np.array_equal(before[:, np.r_[0:24, 27:45]], after[:, np.r_[0:24, 27:45]]):
        raise ValueError("Pose components outside pinky DIP changed")
    if not np.array_equal(timestamps, [f["timestamp_s"] for f in corrected["frames"]]):
        raise ValueError("Timestamps changed")
    import torch
    import smplx
    from smplx.utils import Struct

    torch.set_num_threads(1)
    data = read_mano(Path(source["mano_source"]))
    if digest(Path(source["mano_source"])) != config["mano_sha256"]:
        raise ValueError("MANO model changed after the audit")
    model = smplx.MANO("", data_struct=Struct(**data), is_rhand=False,
                       use_pca=False, flat_hand_mean=True, num_betas=10, dtype=torch.float64)
    perturb = np.zeros((4, 45))
    perturb[1, 24] = perturb[2, 25] = perturb[3, 26] = 0.1
    with torch.no_grad():
        result = model(betas=torch.zeros(4, 10, dtype=torch.float64), hand_pose=torch.tensor(perturb),
                       global_orient=torch.zeros(4, 3, dtype=torch.float64), transl=torch.zeros(4, 3, dtype=torch.float64))
        points = batched_landmarks(result.vertices, result.joints[:, :16]).numpy()
    helper_errors = []
    for i in range(4):
        single, _ = canonical_landmarks(result.vertices[i].numpy(), result.joints[i, :16].numpy())
        helper_errors.append(float(np.max(np.abs(single - points[i]))))
    capture = load_capture(Path(source["source"]), "Left", 0)
    by_id = {f["frame_id"]: f for f in capture["frames"]}
    raw_error = max(float(np.max(np.abs(by_id[f["frame_id"]]["points_local_m"] - np.asarray(f["source_landmarks_m"]))))
                    for f in source["frames"])
    determinants = [np.linalg.det(f["basis_world_from_local"]) for f in capture["frames"]]
    native = {
        "status": "passed_native_model_semantics_check", "expected_left_hand": True,
        "native_parents": data["kintree_table"][0].tolist(), "chains": CHAINS, "tip_vertex_ids": TIP_VERTEX_IDS,
        "tip_skin_weights_at_expected_distal": {name: float(data["weights"][TIP_VERTEX_IDS[name], chain[-1]])
                                               for name, chain in CHAINS.items()},
        "canonical_vs_batched_max_abs_m": max(helper_errors),
        "neutral_forward_vs_template_max_abs_m": float(np.max(np.abs(result.vertices[0].numpy() - data["v_template"]))),
        "pinky_DIP_perturbation_rad": 0.1,
        "native_joint_centers_max_abs_change_m": float(np.max(np.abs(result.joints[1:, :16].numpy() - result.joints[:1, :16].numpy()))),
        "pinky_tip_displacements_m": np.linalg.norm(points[1:, 20] - points[0, 20], axis=1).tolist(),
        "raw_capture_canonicalization_max_abs_m": raw_error,
        "raw_capture_all_frames_physical_left": len(capture["frames"]) == len(source["frames"]),
        "raw_capture_palm_basis_determinant_range": [float(min(determinants)), float(max(determinants))],
        "verify_script_sha256": digest(Path(__file__)),
    }
    if (native["canonical_vs_batched_max_abs_m"] != 0
            or native["native_parents"][7:10] != [0, 7, 8]
            or not all(weight == 1 for weight in native["tip_skin_weights_at_expected_distal"].values())
            or native["native_joint_centers_max_abs_change_m"] > 1e-8
            or native["neutral_forward_vs_template_max_abs_m"] > 1e-7
            or raw_error > 1e-10
            or not np.allclose(determinants, 1, atol=1e-10)):
        raise ValueError("MANO semantic/capture coordinate check failed")
    write_json(run / "native_model_check.json", native)
    temporal = []
    for poses in (before, after):
        rotation = Rotation.from_rotvec(poses[:, 24:27])
        steps = np.degrees((rotation[:-1].inv() * rotation[1:]).magnitude())
        temporal.append({"pinky_DIP_rotation_step_deg": summary(steps),
                         "pinky_DIP_rate_deg_s": summary(steps / np.diff(timestamps)),
                         "max_step_frame_id": int(source["frames"][1 + np.argmax(steps)]["frame_id"]),
                         "max_pose_component_step_rad": float(np.max(np.abs(np.diff(poses, axis=0))))})
    write_json(run / "temporal_check.json", {"status": "completed_temporal_diagnostic_no_filter",
               "before": temporal[0], "after": temporal[1], "timestamps_unchanged": True,
               "note": "SO(3) diagnostics for a noisy visual sequence; no anatomical speed or latency claim."})
    print(json.dumps({"status": native["status"], "raw_capture_canonicalization_max_abs_m": raw_error,
                      "pinky_dip_step_max_deg_before": temporal[0]["pinky_DIP_rotation_step_deg"]["max"],
                      "pinky_dip_step_max_deg_after": temporal[1]["pinky_DIP_rotation_step_deg"]["max"]}))


if __name__ == "__main__":
    main()
