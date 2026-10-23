"""Drive fixed-shape R with a v21 personal MANO sequence (offline baseline).

Source reconstruction includes the fitted scale_i. The target always uses
beta_R/scale_R from its manifest. Only local MANO hand_pose is transferred;
camera/wrist motion is intentionally removed. Both surfaces are anchored by
their MANO wrist at R's neutral wrist, with the manifest's fixed rotation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from fit_mediapipe_to_mano_v21 import batched_landmarks
from run_r_l20_feature_baseline_v15 import make_feature_record, feature_arrays, TIP_PAIRS


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_array(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}, got {array.shape}")
    return array


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--mano", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    import torch
    import smplx
    from smplx.utils import Struct
    from check_mano_model import read_mano

    torch.set_num_threads(min(4, torch.get_num_threads()))
    source_path, model_path = args.input.resolve(), args.model.resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    frames = source["frames"]
    if source.get("hand_side", "").lower() != "left" or not frames:
        raise ValueError("Expected a nonempty v21 physical-left-hand sequence")
    count = len(frames)
    times = finite_array([f["timestamp_s"] for f in frames], (count,), "timestamps")
    if count > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("Timestamps must increase strictly; do not silently reorder/drop frames")
    theta = finite_array([f["theta_hand_pose"] for f in frames], (count, 45), "theta_i")
    source_betas = finite_array([f["beta"] for f in frames], (count, 10), "beta_i")
    if not np.allclose(source_betas, source_betas[0], atol=1e-12, rtol=0):
        raise ValueError("v22 expects one calibrated source shape for the whole sequence")
    source_beta = source_betas[0]
    source_scale = float(source["shape_fit"]["scale_i"])
    if not np.isfinite(source_scale) or source_scale <= 0:
        raise ValueError("scale_i must be positive")
    manifest = json.loads(model_path.read_text(encoding="utf-8"))
    arrays_path = model_path.parent / manifest["arrays"]["npz"]
    with np.load(arrays_path, allow_pickle=False) as arrays:
        target_beta = finite_array(arrays["beta_R"], (10,), "beta_R")
        target_scale = float(arrays["scale"])
        rotation = finite_array(arrays["rotation"], (3, 3), "R rotation")
        translation = finite_array(arrays["translation_m"], (3,), "R translation")
    if target_scale <= 0 or not np.isfinite(target_scale):
        raise ValueError("scale_R must be positive")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or np.linalg.det(rotation) < 0:
        raise ValueError("The manifest rotation must be a proper rotation, not a reflection")
    mano_data = read_mano(args.mano.resolve())
    model = smplx.MANO(str(args.mano.resolve()), data_struct=Struct(**mano_data),
                       is_rhand=False, use_pca=False, flat_hand_mean=True,
                       num_betas=10, dtype=torch.float64)

    def forward(beta, poses, orient=None):
        size = len(poses)
        with torch.no_grad():
            result = model(
                betas=torch.tensor(np.broadcast_to(beta, (size, 10)).copy()),
                hand_pose=torch.tensor(poses),
                global_orient=torch.zeros(size, 3, dtype=torch.float64) if orient is None else torch.tensor(orient),
                transl=torch.zeros(size, 3, dtype=torch.float64),
            )
            points = batched_landmarks(result.vertices, result.joints[:, :16]).numpy()
        return result.vertices.numpy(), points

    _, neutral_target = forward(target_beta, np.zeros((1, 45)))
    anchor = target_scale * (neutral_target[0, 0] @ rotation.T) + translation
    neutral_target_world = target_scale * (neutral_target[0] @ rotation.T) + translation
    all_source_vertices, all_r_vertices = [], []
    all_source_points, all_r_points = [], []
    reconstruction_errors = []
    for start in range(0, count, args.batch_size):
        stop = min(start + args.batch_size, count)
        poses = theta[start:stop]
        for beta, scale, vertex_out, point_out in (
                (source_beta, source_scale, all_source_vertices, all_source_points),
                (target_beta, target_scale, all_r_vertices, all_r_points)):
            vertices, points = forward(beta, poses)
            wrist = points[:, :1, :]
            vertex_out.append(scale * ((vertices - wrist) @ rotation.T) + anchor)
            point_out.append(scale * ((points - wrist) @ rotation.T) + anchor)
        # Check that the source reconstruction honors v21's scale/rotation/
        # translation convention before discarding global motion for control.
        orientations = np.asarray([f["global_orient"] for f in frames[start:stop]])
        translations = np.asarray([f["transl"] for f in frames[start:stop]])
        _, reconstructed = forward(source_beta, poses, orientations)
        reconstructed = source_scale * reconstructed + translations[:, None, :]
        stored = np.asarray([f["fitted_landmarks_m"] for f in frames[start:stop]])
        reconstruction_errors.append(np.max(np.abs(reconstructed - stored)))
        print(json.dumps({"status": "transferring", "frames": stop, "total": count}), flush=True)
    source_vertices, r_vertices = np.concatenate(all_source_vertices), np.concatenate(all_r_vertices)
    source_points, r_points = np.concatenate(all_source_points), np.concatenate(all_r_points)
    max_reconstruction_error = float(max(reconstruction_errors))
    if max_reconstruction_error > 1e-7:
        raise ValueError(f"Source convention reconstruction failed: {max_reconstruction_error} m")

    direction_errors, tip_errors, tip_source, tip_r = [], [], [], []
    frame_metrics = []
    for i, (sp, rp) in enumerate(zip(source_points, r_points)):
        sf, rf = feature_arrays(make_feature_record(sp)), feature_arrays(make_feature_record(rp))
        dot = np.clip(np.sum(sf["directions"] * rf["directions"], axis=1), -1, 1)
        direction = np.degrees(np.arccos(dot))
        tip_error = np.abs(sf["tip_distances"] - rf["tip_distances"])
        direction_errors.append(direction)
        tip_errors.append(tip_error)
        tip_source.append(sf["tip_distances"])
        tip_r.append(rf["tip_distances"])
        frame_metrics.append({"frame_id": int(frames[i]["frame_id"]),
                              "bone_direction_rmse_deg": float(np.sqrt(np.mean(direction**2))),
                              "mean_tip_pair_error_normalized": float(tip_error.mean())})
    direction_errors, tip_errors = np.asarray(direction_errors), np.asarray(tip_errors)
    tip_source, tip_r = np.asarray(tip_source), np.asarray(tip_r)
    ids = np.asarray([int(f["frame_id"]) for f in frames])
    pair_metrics = {}
    selected_indices = {0, count // 2, count - 1}
    for pair_index, pair in enumerate(TIP_PAIRS):
        near = int(np.argmin(tip_source[:, pair_index]))
        if pair in [("thumb_tip", "index_tip"), ("thumb_tip", "middle_tip")]:
            selected_indices.add(near)
        pair_metrics["+".join(pair)] = {
            "mean_abs_distance_error_normalized": float(tip_errors[:, pair_index].mean()),
            "max_abs_distance_error_normalized": float(tip_errors[:, pair_index].max()),
            "source_closest_frame_id": int(ids[near]),
            "source_distance_normalized_at_source_closest": float(tip_source[near, pair_index]),
            "r_distance_normalized_at_source_closest": float(tip_r[near, pair_index]),
        }
    source_record = make_feature_record(source_points[0])
    target_record = make_feature_record(r_points[0])
    normal = np.asarray(target_record["palm_basis_root_columns"])[:, 1]
    # View the palm from the same approximate side as earlier L20 previews.
    if normal[0] > 0:
        normal = -normal

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "motion_geometry.npz",
                        source_vertices_m=source_vertices.astype(np.float32),
                        r_vertices_m=r_vertices.astype(np.float32),
                        source_points_m=source_points, r_points_m=r_points,
                        faces=np.asarray(mano_data["f"], dtype=np.int32),
                        timestamps_s=times, frame_ids=ids,
                        camera_normal=normal, neutral_r_points_m=neutral_target_world)
    output_frames = []
    for i, frame in enumerate(frames):
        output_frames.append({
            "frame_id": int(ids[i]), "timestamp_s": float(times[i]),
            "name": frame.get("name", f"frame_{ids[i]:04d}"),
            "phase": "v22_personal_to_fixed_R", "kind": "fixed_R_direct_local_pose_transfer",
            "theta_hand_pose": theta[i].tolist(), "beta": target_beta.tolist(),
            "global_orient": [0.0] * 3, "transl": [0.0] * 3,
            "target_landmarks_m": r_points[i].tolist(),
            "source_beta_i": source_beta.tolist(),
            "source_scale_i": source_scale,
            "source_reference_landmarks_m": source_points[i].tolist(),
        })
    limitations = [
        "Offline playback of one operator; not live camera control or cross-operator evaluation.",
        "Direct local MANO pose transfer only; no task/contact optimization or learned network.",
        "R beta_R and scale_R are fixed from the v12 candidate; this surface is not an exact L20 mesh.",
        "Physical wrist/camera translation and orientation are not transferred; wrists share a fixed anchor.",
        "Source metric scale is the MediaPipe/MANO estimate, not independent hand-size metrology.",
        "No L20 q/mimic/u optimization, collision or contact is performed.",
        "Use target_landmarks_m as authoritative for v22; v20 re-forward would omit wrist anchoring.",
    ]
    payload = {
        "version": "v22_personal_hand_to_fixed_R", "status": "completed_offline_direct_pose_transfer",
        "source": str(source_path), "model": str(model_path),
        "hand_side": "left", "frame_count": count,
        "units": {"landmarks": "m", "theta": "rad", "timestamp": "s"},
        "target_geometry": {"beta_R": target_beta.tolist(), "scale_R": target_scale,
                            "rotation": rotation.tolist(), "wrist_anchor_m": anchor.tolist()},
        "normalization": {"beta_source": "fixed manifest beta_R, never input beta_i",
                          "pose_mapping": "theta_R(t) = theta_i(t)",
                          "wrist_policy": "source and R independently wrist-centered at fixed R anchor"},
        "frames": output_frames, "limitations": limitations,
    }
    (output / "r_sequence_v22.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metrics = {
        "version": "v22_personal_hand_to_fixed_R", "status": payload["status"],
        "frame_count": count, "duration_s": float(times[-1] - times[0]),
        "source_beta_i": source_beta.tolist(), "target_beta_R": target_beta.tolist(),
        "source_scale_i": source_scale, "target_scale_R": target_scale,
        "source_reconstruction_max_abs_error_m": max_reconstruction_error,
        "target_beta_fixed": bool(np.allclose(np.asarray([f["beta"] for f in output_frames]), target_beta, atol=0, rtol=0)),
        "local_pose_transfer_max_abs_error_rad": float(np.max(np.abs(np.asarray([f["theta_hand_pose"] for f in output_frames]) - theta))),
        "wrist_anchor_max_abs_error_m": float(np.max(np.abs(r_points[:, 0] - anchor))),
        "bone_direction_rmse_deg": float(np.sqrt(np.mean(direction_errors**2))),
        "tip_pair_mean_abs_error_normalized": float(tip_errors.mean()),
        "tip_pairs": pair_metrics, "diagnostic_frame_ids": ids[sorted(selected_indices)].tolist(),
        "source_first_frame_palm_scale_m": source_record["palm_scale_m"],
        "r_first_frame_palm_scale_m": target_record["palm_scale_m"],
        "pose_step_max_abs_rad": float(np.max(np.abs(np.diff(theta, axis=0)))) if count > 1 else 0.0,
        "pose_parameter_rate_component_abs_p95_rad_s": float(np.percentile(np.abs(np.diff(theta, axis=0) / np.diff(times)[:, None]), 95)) if count > 1 else 0.0,
        "frame_metrics": frame_metrics, "limitations": limitations,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    config = {"version": payload["version"], "input": str(source_path),
              "input_sha256": sha256(source_path), "model": str(model_path),
              "model_arrays_sha256": sha256(arrays_path), "script_sha256": sha256(Path(__file__)),
              "mano": str(args.mano.resolve()), "point_order": "MediaPipe-compatible 21 semantic MANO points",
              "evaluation": "single-operator diagnostic; all input frames, all 20 bones and 10 tip pairs",
              "no_frame_drop": True, "target_wrist_anchor_m": anchor.tolist(),
              "mapping": "direct local pose transfer; fixed beta_R/scale_R; no extra smoothing"}
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ("status", "frame_count", "duration_s", "bone_direction_rmse_deg", "tip_pair_mean_abs_error_normalized", "source_reconstruction_max_abs_error_m")}), flush=True)


if __name__ == "__main__":
    main()
