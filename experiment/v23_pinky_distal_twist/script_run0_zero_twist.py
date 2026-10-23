"""Audit and remove the unobserved pinky DIP twist in a v21 sequence.

Only local MANO joint 9 (hand_pose[24:27]) changes. Shape, scale, all other
rotations, root transform and timestamps are frozen. The candidate uses a
zero-twist swing rotation about the shape-specific neutral DIP-to-tip axis.
Its two swing coordinates are fitted to the ORIGINAL fitted 21 points, not
to extra labels. This is a neutral-twist prior, not recovered nail truth.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from check_mano_model import CHAINS, TIP_VERTEX_IDS, read_mano
from fit_mediapipe_to_mano_v21 import BONE_EDGES, batched_landmarks


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unit(values):
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norms < 1e-10):
        raise ValueError("Degenerate direction")
    return values / norms


def angle(a, b):
    return np.degrees(np.arccos(np.clip(np.sum(unit(a) * unit(b), axis=-1), -1, 1)))


def twist_deg(rotvec, axis):
    quaternion = Rotation.from_rotvec(rotvec).as_quat()
    twist = 2 * np.arctan2(quaternion[:, :3] @ axis, quaternion[:, 3])
    return np.degrees((twist + np.pi) % (2 * np.pi) - np.pi)


def neutral_swing(rotvec, axis):
    """Right swing-twist split: R = swing * twist, local neutral bone axis."""
    quaternion = Rotation.from_rotvec(rotvec).as_quat()
    twist = np.column_stack([(quaternion[:, :3] @ axis)[:, None] * axis, quaternion[:, 3]])
    norm = np.linalg.norm(twist, axis=1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("Ambiguous 180-degree swing; do not choose a twist silently")
    swing = Rotation.from_quat(quaternion) * Rotation.from_quat(twist / norm).inv()
    return swing.as_rotvec()


def summary(values):
    return {"mean": float(np.mean(values)), "rmse": float(np.sqrt(np.mean(values ** 2))),
            "max": float(np.max(values)), "p95": float(np.percentile(values, 95))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("experiment/v21_MediaPipe到MANO序列拟合/my_mano_sequence_v21_all.json"))
    parser.add_argument("--motion", type=Path, default=Path("experiment/v22_personal_fixed_r/motion_geometry.npz"))
    parser.add_argument("--manifest", type=Path, default=Path("experiment/v12_R_L20_canonical_human/r_l20_model.json"))
    parser.add_argument("--mano", type=Path, default=Path("E:/dex/MANO/mano_v1_2/mano_v1_2/models/MANO_LEFT.pkl"))
    parser.add_argument("--output", type=Path, default=Path("experiment/v23_pinky_distal_twist"))
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.json").exists():
        raise FileExistsError("Preserve previous metrics; use a new output directory")
    import torch
    import smplx
    from smplx.utils import Struct

    torch.set_num_threads(2)
    source = json.loads(args.input.read_text(encoding="utf-8"))
    frames = source["frames"]
    count = len(frames)
    if source["hand_side"].lower() != "left" or count < 2:
        raise ValueError("Expected physical-left sequence with at least two frames")
    poses = np.asarray([f["theta_hand_pose"] for f in frames], dtype=float)
    betas = np.asarray([f["beta"] for f in frames], dtype=float)
    times = np.asarray([f["timestamp_s"] for f in frames])
    if poses.shape != (count, 45) or not np.isfinite(poses).all():
        raise ValueError("Expected finite 45D radian poses")
    if not np.allclose(betas, betas[0], atol=1e-12, rtol=0) or np.any(np.diff(times) <= 0):
        raise ValueError("Expected fixed source shape and increasing timestamps")
    beta = betas[0]
    scale = float(source["shape_fit"]["scale_i"])
    observed = np.asarray([f["source_landmarks_m"] for f in frames])
    fitted = np.asarray([f["fitted_landmarks_m"] for f in frames])
    global_rotations = Rotation.from_rotvec([f["global_orient"] for f in frames]).as_matrix()
    translations = np.asarray([f["transl"] for f in frames])
    data = read_mano(args.mano.resolve())
    model = smplx.MANO(str(args.mano.resolve()), data_struct=Struct(**data),
                       is_rhand=False, use_pca=False, flat_hand_mean=True,
                       num_betas=10, dtype=torch.float64)

    def forward(current, shape=beta):
        n = len(current)
        with torch.no_grad():
            result = model(betas=torch.tensor(np.broadcast_to(shape, (n, 10)).copy()),
                           hand_pose=torch.tensor(current, dtype=torch.float64),
                           global_orient=torch.zeros(n, 3, dtype=torch.float64),
                           transl=torch.zeros(n, 3, dtype=torch.float64))
            points = batched_landmarks(result.vertices, result.joints[:, :16]).numpy()
        return result.vertices.numpy(), points

    def chunks(current, shape=beta):
        vertices, points = [], []
        for start in range(0, len(current), 64):
            v, p = forward(current[start:start + 64], shape)
            vertices.append(v)
            points.append(p)
        return np.concatenate(vertices), np.concatenate(points)

    neutral_vertices, neutral_points = forward(np.zeros((1, 45)))
    axis = unit(neutral_points[0, 20] - neutral_points[0, 19])
    parents = data["kintree_table"][0].astype(np.int64)
    expected = [0, 7, 8]
    if parents[7:10].tolist() != expected or int(np.argmax(data["weights"][671])) != 9:
        raise ValueError("Pinky chain/tip skin assignment does not match this audit")
    # A tangent basis parameterizes swing only. Its rotvec is perpendicular
    # to the neutral distal axis, so it has zero axial twist by construction.
    helper = np.eye(3)[int(np.argmin(np.abs(axis)))]
    basis = np.column_stack([unit(np.cross(axis, helper)), np.zeros(3)])
    basis[:, 1] = unit(np.cross(axis, basis[:, 0]))
    original_vertices, original_points = chunks(poses)
    # MANO global_orient rotates about its native wrist joint, not the
    # coordinate origin. Translation is applied after the metric scale.
    wrist = original_points[:, :1]
    reconstruction = scale * (np.einsum("nkj,nij->nki", original_points - wrist, global_rotations) + wrist) + translations[:, None]
    if np.max(np.abs(reconstruction - fitted)) > 1e-8:
        raise ValueError("Source reconstruction does not reproduce the v21 point contract")

    corrected = poses.copy()
    initial_swing = neutral_swing(poses[:, 24:27], axis)
    corrected[:, 24:27] = initial_swing
    unrefined_vertices, unrefined_points = chunks(corrected)
    histories = []
    # Independent 2-variable least squares for every frame. Batched central
    # finite differences keep the Jacobians independent across frames.
    # Damping and step cap are numerical settings, not anatomical thresholds.
    for start in range(0, count, 64):
        stop = min(start + 64, count)
        coordinates = initial_swing[start:stop] @ basis
        template = poses[start:stop].copy()
        target = original_points[start:stop]

        def residual(x):
            current = template.copy()
            current[:, 24:27] = x @ basis.T
            _, points = forward(current)
            return (points - target).reshape(len(x), -1)

        old_cost = np.sum(residual(coordinates) ** 2, axis=1)
        for iteration in range(18):
            r = residual(coordinates)
            columns = []
            for column in range(2):
                perturb = np.zeros_like(coordinates)
                perturb[:, column] = 1e-5
                columns.append((residual(coordinates + perturb) - residual(coordinates - perturb)) / 2e-5)
            jac = np.stack(columns, axis=2)
            hessian = np.einsum("nki,nkj->nij", jac, jac) + 1e-9 * np.eye(2)[None]
            gradient = np.einsum("nki,nk->ni", jac, r)
            step = -np.linalg.solve(hessian, gradient[..., None])[..., 0]
            step /= np.maximum(1, np.linalg.norm(step, axis=1, keepdims=True) / 0.2)
            # Per-frame backtracking; frames cannot exchange updates.
            for multiplier in (1.0, 0.5, 0.25, 0.125):
                trial = coordinates + multiplier * step
                cost = np.sum(residual(trial) ** 2, axis=1)
                take = cost < old_cost
                coordinates[take] = trial[take]
                old_cost[take] = cost[take]
                step[take] = 0
        corrected[start:stop, 24:27] = coordinates @ basis.T
        histories.append({"start": start, "stop_exclusive": stop, "iterations": iteration + 1,
                          "point_residual_sum_m2": float(old_cost.sum())})
        print(json.dumps({"status": "v23_swing_refined", "frames": stop, "total": count}), flush=True)

    new_vertices, new_points = chunks(corrected)
    new_wrist = new_points[:, :1]
    new_fitted = scale * (np.einsum("nkj,nij->nki", new_points - new_wrist, global_rotations) + new_wrist) + translations[:, None]
    new_errors = np.linalg.norm(new_fitted - observed, axis=2) * 1000
    old_errors = np.linalg.norm(fitted - observed, axis=2) * 1000
    before_twist = twist_deg(poses[:, 24:27], axis)
    after_twist = twist_deg(corrected[:, 24:27], axis)
    tip_change = np.linalg.norm(new_fitted[:, 20] - fitted[:, 20], axis=1) * 1000
    tip_direction_change = angle(new_fitted[:, 20] - new_fitted[:, 19], fitted[:, 20] - fitted[:, 19])
    old_direction = angle(fitted[:, 20] - fitted[:, 19], observed[:, 20] - observed[:, 19])
    new_direction = angle(new_fitted[:, 20] - new_fitted[:, 19], observed[:, 20] - observed[:, 19])
    old_bones = np.linalg.norm(fitted[:, BONE_EDGES[:, 1]] - fitted[:, BONE_EDGES[:, 0]], axis=2)
    new_bones = np.linalg.norm(new_fitted[:, BONE_EDGES[:, 1]] - new_fitted[:, BONE_EDGES[:, 0]], axis=2)
    obs_bones = np.linalg.norm(observed[:, BONE_EDGES[:, 1]] - observed[:, BONE_EDGES[:, 0]], axis=2)
    frame_ids = np.asarray([f["frame_id"] for f in frames])
    fixed_ids = [0, 190, 383, 773]
    added_ids = [5, 30, 60, 120, 250]
    config = {
        "version": "v23_pinky_distal_twist", "input": str(args.input.resolve()),
        "input_sha256": digest(args.input), "mano_sha256": digest(args.mano),
        "script_sha256": digest(Path(__file__)), "manifest_sha256": digest(args.manifest),
        "baseline_geometry_sha256": digest(args.motion), "units": {"points": "m", "pose": "rad", "angles_reported": "deg"},
        "hand_side": "left", "target": "original v21 fitted 21 points, fixed shape and root",
        "changed_pose_slice": [24, 27], "native_joint": 9, "semantic_joint": "pinky DIP",
        "twist_axis": "shape-specific neutral DIP-to-tip (includes fingertip surface offset)",
        "source_neutral_axis": axis.tolist(), "swing_basis": basis.tolist(),
        "prior": "zero axial twist relative to MANO neutral; nail orientation is not observed",
        "least_squares": {"batch_size": 64, "iterations_max": 18, "finite_difference_rad": 1e-5,
                          "damping": 1e-9, "step_cap_rad": 0.2, "line_search": [1, 0.5, 0.25, 0.125]},
        "evaluation": "all input frames; diagnostic only, no train/test or generalization claim",
        "initial_window": "first min(60,N) frames by sequence index; descriptive, not acceptance threshold",
        "existing_diagnostic_frame_ids": fixed_ids, "added_failure_frame_ids": added_ids,
    }
    write_json(out / "config.json", config)
    metrics = {
        "version": config["version"], "status": "completed_offline_model_audit_and_neutral_twist_candidate",
        "frame_count": count, "duration_s": float(times[-1] - times[0]),
        "model_audit": {"pinky_native_joints": [7, 8, 9], "pinky_parents": parents[7:10].tolist(),
                        "pinky_tip_vertex": 671, "tip_skin_joint": 9, "tip_skin_weight": float(data["weights"][671, 9]),
                        "source_reconstruction_max_abs_m": float(np.max(np.abs(reconstruction - fitted))),
                        "old_mujoco_display": "v22 full-frame scene_check; not changed by this script"},
        "unchanged_pose_components_max_abs_rad": float(np.max(np.abs(corrected[:, np.r_[0:24, 27:45]] - poses[:, np.r_[0:24, 27:45]]))),
        "source_pinky_dip_twist_abs_deg_before": summary(np.abs(before_twist)),
        "source_pinky_dip_twist_abs_deg_after": summary(np.abs(after_twist)),
        "initial60_twist_abs_deg_before": summary(np.abs(before_twist[:60])),
        "pinky_tip_change_from_original_fit_mm": summary(tip_change),
        "pinky_tip_direction_change_from_original_fit_deg": summary(tip_direction_change),
        "all21_observation_point_error_mm_before": summary(old_errors),
        "all21_observation_point_error_mm_after": summary(new_errors),
        "pinky_tip_observation_error_mm_before": summary(old_errors[:, 20]),
        "pinky_tip_observation_error_mm_after": summary(new_errors[:, 20]),
        "bone_length_mae_to_observation_mm_before": float(np.mean(np.abs(old_bones - obs_bones)) * 1000),
        "bone_length_mae_to_observation_mm_after": float(np.mean(np.abs(new_bones - obs_bones)) * 1000),
        "pinky_tip_direction_to_observation_deg_before": summary(old_direction),
        "pinky_tip_direction_to_observation_deg_after": summary(new_direction),
        "simple_twist_removal_tip_change_mm": summary(scale * np.linalg.norm(unrefined_points[:, 20] - original_points[:, 20], axis=1) * 1000),
        "diagnostic_frames": [
            {"frame_id": int(frame_ids[i]), "relative_s": float(times[i] - times[0]),
             "twist_deg_before": float(before_twist[i]), "twist_deg_after": float(after_twist[i]),
             "tip_change_mm": float(tip_change[i]), "tip_direction_change_deg": float(tip_direction_change[i]),
             "point_tip_error_mm_before": float(old_errors[i, 20]), "point_tip_error_mm_after": float(new_errors[i, 20])}
            for i in range(count) if int(frame_ids[i]) in fixed_ids + added_ids],
        "solver_history": histories,
        "limitations": ["Neutral-twist prior, not ground-truth nail orientation or measured anatomy.",
                        "Only pinky DIP is corrected; other finger twists are not addressed.",
                        "No new shape, camera capture, temporal filter, network, L20 q/u or physics.",
                        "Single-operator offline diagnostic; observation errors are relative to MediaPipe."],
    }
    write_json(out / "metrics.json", metrics)
    payload = copy.deepcopy(source)
    payload["version"] = config["version"]
    payload["status"] = metrics["status"]
    payload["v23_input_sequence"] = str(args.input.resolve())
    payload["original_motion_fit_v21"] = payload.pop("motion_fit")
    payload["postprocess_v23"] = {"config": "config.json", "prior": config["prior"], "changed_pose_slice": [24, 27]}
    payload["motion_fit"] = {"point_rmse_mm": metrics["all21_observation_point_error_mm_after"]["rmse"],
                            "point_max_mm": float(new_errors.max()),
                            "bone_length_mae_mm": metrics["bone_length_mae_to_observation_mm_after"],
                            "note": "v23 postprocess; original v21 optimizer metrics retained separately"}
    for i, frame in enumerate(payload["frames"]):
        frame["theta_hand_pose"] = corrected[i].tolist()
        frame["fitted_landmarks_m"] = new_fitted[i].tolist()
        frame["point_error_mm"] = new_errors[i].tolist()
        frame["phase"] = config["version"]
    write_json(out / "mano_sequence_v23.json", payload)
    np.savez_compressed(out / "audit_geometry.npz", original_vertices=original_vertices,
                        new_vertices=new_vertices, original_points=original_points,
                        new_points=new_points, observed_points=observed, old_fitted=fitted,
                        new_fitted=new_fitted, faces=data["f"], source_neutral_vertices=neutral_vertices[0],
                        source_neutral_points=neutral_points[0], before_twist_deg=before_twist,
                        after_twist_deg=after_twist, frame_ids=frame_ids, timestamps_s=times,
                        source_beta=beta, new_poses=corrected)
    print(json.dumps({key: metrics[key] for key in ("status", "frame_count", "initial60_twist_abs_deg_before",
                      "pinky_tip_change_from_original_fit_mm", "all21_observation_point_error_mm_before",
                      "all21_observation_point_error_mm_after")}), flush=True)


if __name__ == "__main__":
    main()
