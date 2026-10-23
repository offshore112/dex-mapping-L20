"""Fit a shared per-operator MANO shape and a temporal MANO pose sequence.

This is v21's first offline baseline.  It consumes the MediaPipe JSON written
by ``capture_camera_mediapipe_v20.py`` and never treats MediaPipe landmarks as
MANO beta/theta or as L20 joint targets.

The fitting is deliberately two-stage:

1. Use an evenly sampled calibration subset to estimate one beta vector and
   one metric scale for the operator.  Per-calibration-frame pose, orientation
   and translation absorb pose variation, while shape is shared.
2. Freeze beta and scale, then fit a pose/orientation/translation for every
   selected frame with bone-length, pose-prior and temporal-smoothness terms.

The output contains ``theta_hand_pose``, ``beta``, ``global_orient`` and
``transl`` fields compatible with ``prepare_offline_mano_sequence_v20.py``.
The fitting frame is a wrist-centered, palm-aligned frame.  The original
MediaPipe wrist translation is preserved separately as metadata and is not
silently mixed into MANO's canonical pose.

This script is an offline diagnostic baseline.  It does not perform the
shared-R mapping or L20 q/mimic/actuator mapping.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from check_mano_model import read_mano  # noqa: E402
from fit_mano_r import mano_landmarks_torch  # noqa: E402


LANDMARK_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]

# MediaPipe order is wrist, thumb(1-4), index(5-8), middle(9-12),
# ring(13-16), pinky(17-20), which is the same semantic order used by the
# project's MANO landmark helper.
BONE_EDGES = np.asarray([
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12],
    [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20],
], dtype=int)


def _unit(vector: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < eps:
        raise ValueError("degenerate palm frame")
    return vector / norm


def canonicalize_mediapipe(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return wrist-centered palm coordinates and the world-frame basis.

    The x axis points from pinky MCP to index MCP, the y axis points from
    wrist to middle MCP, and z completes a right-handed frame.  No reflection
    is applied; physical hand side is checked before this function is called.
    """
    points = np.asarray(points, dtype=float)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        raise ValueError("landmarks must be a finite (21, 3) array")
    wrist = points[0]
    x_axis = _unit(points[5] - points[17])
    palm_axis = points[9] - wrist
    palm_axis = palm_axis - np.dot(palm_axis, x_axis) * x_axis
    y_axis = _unit(palm_axis)
    z_axis = _unit(np.cross(x_axis, y_axis))
    y_axis = _unit(np.cross(z_axis, x_axis))
    basis = np.column_stack([x_axis, y_axis, z_axis])
    local = (points - wrist) @ basis
    return local, basis


def weighted_similarity(source: np.ndarray, target: np.ndarray,
                        weights: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit target ~= scale * source @ rotation.T + translation without reflection."""
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    source_center = (source * weights[:, None]).sum(axis=0)
    target_center = (target * weights[:, None]).sum(axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    covariance = (source_zero * weights[:, None]).T @ target_zero
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    rotated = source_zero @ rotation.T
    numerator = float((weights[:, None] * rotated * target_zero).sum())
    denominator = float((weights[:, None] * source_zero * source_zero).sum())
    scale = max(numerator / max(denominator, 1e-12), 1e-8)
    translation = target_center - scale * (rotation @ source_center)
    return scale, rotation, translation


def rotation_matrix_to_axis_angle(rotation: np.ndarray) -> np.ndarray:
    value = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(value))
    if angle < 1e-8:
        return 0.5 * np.asarray([
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ])
    axis = np.asarray([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ]) / (2.0 * np.sin(angle))
    return axis * angle


def batched_landmarks(vertices, joints):
    """Batched version of the project's MANO 21-point helper."""
    import torch

    tip_ids = [744, 320, 443, 554, 671]
    pieces = [joints[:, 0:1, :]]
    pieces.extend([joints[:, 13:16, :], vertices[:, tip_ids[0]:tip_ids[0] + 1, :]])
    pieces.extend([joints[:, 1:4, :], vertices[:, tip_ids[1]:tip_ids[1] + 1, :]])
    pieces.extend([joints[:, 4:7, :], vertices[:, tip_ids[2]:tip_ids[2] + 1, :]])
    pieces.extend([joints[:, 10:13, :], vertices[:, tip_ids[3]:tip_ids[3] + 1, :]])
    pieces.extend([joints[:, 7:10, :], vertices[:, tip_ids[4]:tip_ids[4] + 1, :]])
    return torch.cat(pieces, dim=1)


def load_capture(path: Path, hand_side: str, max_frames: int) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_frames = payload.get("frames")
    if not isinstance(raw_frames, list) or not raw_frames:
        raise ValueError("input JSON must contain a non-empty frames list")
    selected = []
    skipped = {"missing_world_landmarks": 0, "wrong_handedness": 0,
               "invalid_landmarks": 0}
    for raw in raw_frames:
        if hand_side and raw.get("handedness") != hand_side:
            skipped["wrong_handedness"] += 1
            continue
        values = raw.get("world_landmarks_m")
        if values is None:
            skipped["missing_world_landmarks"] += 1
            continue
        try:
            points = np.asarray(values, dtype=float)
            local, basis = canonicalize_mediapipe(points)
        except (TypeError, ValueError):
            skipped["invalid_landmarks"] += 1
            continue
        selected.append({
            "frame_id": int(raw.get("frame_id", len(selected))),
            "timestamp_s": float(raw.get("timestamp_s", len(selected) / 30.0)),
            "points_world_m": points,
            "points_local_m": local,
            "wrist_world_m": points[0],
            "basis_world_from_local": basis,
        })
    if not selected:
        raise ValueError(f"no usable {hand_side or 'hand'} world-landmark frames; skipped={skipped}")
    if max_frames > 0 and len(selected) > max_frames:
        indices = np.linspace(0, len(selected) - 1, max_frames).round().astype(int)
        indices = np.unique(indices)
        selected = [selected[int(i)] for i in indices]
    return {"source": payload, "frames": selected, "skipped": skipped}


def evenly_spaced_indices(count: int, maximum: int) -> np.ndarray:
    if count <= 0:
        raise ValueError("count must be positive")
    if maximum <= 0 or count <= maximum:
        return np.arange(count, dtype=int)
    return np.unique(np.linspace(0, count - 1, maximum).round().astype(int))


def morphology_summary(points: np.ndarray) -> dict:
    lengths = np.linalg.norm(points[:, BONE_EDGES[:, 1]] - points[:, BONE_EDGES[:, 0]], axis=2)
    median_bones = np.median(lengths, axis=0)
    chains = {
        "thumb": [0, 1, 2, 3],
        "index": [4, 5, 6, 7],
        "middle": [8, 9, 10, 11],
        "ring": [12, 13, 14, 15],
        "pinky": [16, 17, 18, 19],
    }
    finger_lengths = {name: float(median_bones[ids].sum()) for name, ids in chains.items()}
    palm_length = np.linalg.norm(points[:, 9] - points[:, 0], axis=1)
    palm_width = np.linalg.norm(points[:, 5] - points[:, 17], axis=1)
    return {
        "palm_length_m_median": float(np.median(palm_length)),
        "palm_width_m_median": float(np.median(palm_width)),
        "finger_length_m_median": finger_lengths,
        "bone_length_m_median": median_bones.tolist(),
        "bone_edges": BONE_EDGES.tolist(),
    }


def _initial_pose(model, targets: np.ndarray, neutral: np.ndarray,
                  weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scales, rotations, translations = [], [], []
    for target in targets:
        scale, rotation, translation = weighted_similarity(neutral, target, weights)
        scales.append(scale)
        rotations.append(rotation_matrix_to_axis_angle(rotation))
        translations.append(translation)
    return (np.asarray(rotations, dtype=float),
            np.asarray(translations, dtype=float),
            np.asarray(scales, dtype=float))


def _forward(model, beta, pose_raw, orient_raw, translation, log_scale,
             beta_limit: float, pose_limit: float):
    import torch

    beta_value = beta_limit * torch.tanh(beta)
    pose_value = pose_limit * torch.tanh(pose_raw)
    orient_value = pose_limit * torch.tanh(orient_raw)
    scale_value = torch.exp(log_scale)
    batch = pose_value.shape[0]
    output = model(
        betas=beta_value.unsqueeze(0).expand(batch, -1),
        hand_pose=pose_value,
        global_orient=orient_value,
        transl=torch.zeros_like(translation),
    )
    points = scale_value * batched_landmarks(output.vertices, output.joints[:, :16, :]) + translation[:, None, :]
    return points, beta_value, pose_value, orient_value, scale_value


def _objective(model, target, target_bones, weights, d_ref, beta, pose_raw,
               orient_raw, translation, log_scale, beta_limit, pose_limit,
               optimize_shape, previous=None, *, shape_prior, pose_prior,
               bone_weight, smooth_weight, translation_prior):
    import torch

    predicted, beta_value, pose_value, orient_value, scale_value = _forward(
        model, beta, pose_raw, orient_raw, translation, log_scale,
        beta_limit, pose_limit)
    weights_t = torch.as_tensor(weights, dtype=target.dtype)
    point_loss = (weights_t[None, :, None] * (predicted - target).square()).sum()
    point_loss = point_loss / (weights_t.sum() * target.shape[0] * d_ref * d_ref)
    edge_i = torch.as_tensor(BONE_EDGES[:, 0], dtype=torch.long)
    edge_j = torch.as_tensor(BONE_EDGES[:, 1], dtype=torch.long)
    predicted_bones = torch.linalg.norm(predicted[:, edge_j] - predicted[:, edge_i], dim=2)
    bone_loss = (predicted_bones - target_bones).square().mean() / (d_ref * d_ref)
    shape_loss = beta_value.square().mean()
    pose_loss = pose_value.square().mean()
    trans_loss = translation.square().mean() / (d_ref * d_ref)
    smooth_loss = torch.zeros_like(point_loss)
    if predicted.shape[0] > 1:
        smooth_loss = (
            (pose_value[1:] - pose_value[:-1]).square().mean()
            + 0.25 * (orient_value[1:] - orient_value[:-1]).square().mean()
            + 0.10 * (translation[1:] - translation[:-1]).square().mean() / (d_ref * d_ref)
        )
    total = point_loss + bone_weight * bone_loss + shape_prior * shape_loss
    total = total + pose_prior * pose_loss + translation_prior * trans_loss
    total = total + smooth_weight * smooth_loss
    if not optimize_shape:
        total = point_loss + bone_weight * bone_loss + pose_prior * pose_loss
        total = total + translation_prior * trans_loss + smooth_weight * smooth_loss
    return total, {
        "point_loss": point_loss,
        "bone_loss": bone_loss,
        "shape_loss": shape_loss,
        "pose_loss": pose_loss,
        "translation_loss": trans_loss,
        "smooth_loss": smooth_loss,
        "predicted": predicted,
        "beta_value": beta_value,
        "pose_value": pose_value,
        "orient_value": orient_value,
        "scale_value": scale_value,
    }


def optimize_batch(mano_path: Path, targets: np.ndarray, *, beta_init=None,
                   scale_init=None, orient_init=None, translation_init=None,
                   pose_init=None, steps: int, learning_rate: float,
                   beta_limit: float, pose_limit: float, shape_prior: float,
                   pose_prior: float, bone_weight: float, smooth_weight: float,
                   translation_prior: float, optimize_shape: bool) -> dict:
    import smplx
    import torch
    from smplx.utils import Struct

    torch.set_num_threads(1)
    dtype = torch.float64
    frame_count = int(targets.shape[0])
    data = read_mano(mano_path.resolve())
    model = smplx.MANO(str(mano_path.resolve()), data_struct=Struct(**data),
                       is_rhand=False, use_pca=False, flat_hand_mean=True,
                       num_betas=10, dtype=dtype, batch_size=frame_count)
    with torch.no_grad():
        neutral_out = model(
            betas=torch.zeros(frame_count, 10, dtype=dtype),
            hand_pose=torch.zeros(frame_count, 45, dtype=dtype),
            global_orient=torch.zeros(frame_count, 3, dtype=dtype),
            transl=torch.zeros(frame_count, 3, dtype=dtype),
        )
        neutral = batched_landmarks(neutral_out.vertices, neutral_out.joints[:, :16, :])[0].cpu().numpy()
    weights = np.ones(21, dtype=float)
    weights[0] = 2.0
    orient0, translation0, scales0 = _initial_pose(model, targets, neutral, weights)
    if orient_init is not None:
        orient0 = np.asarray(orient_init, dtype=float)
    if translation_init is not None:
        translation0 = np.asarray(translation_init, dtype=float)
    if pose_init is None:
        pose0 = np.zeros((frame_count, 45), dtype=float)
    else:
        pose0 = np.asarray(pose_init, dtype=float)
    if scale_init is None:
        scale0 = float(np.median(scales0))
    else:
        scale0 = float(scale_init)
    if beta_init is None:
        beta0 = np.zeros(10, dtype=float)
    else:
        beta0 = np.asarray(beta_init, dtype=float)
    d_ref = float(np.median(np.linalg.norm(targets[:, 9] - targets[:, 0], axis=1)))
    d_ref = max(d_ref, 1e-4)
    edge_i, edge_j = BONE_EDGES[:, 0], BONE_EDGES[:, 1]
    target_bones_np = np.linalg.norm(targets[:, edge_j] - targets[:, edge_i], axis=2)
    target = torch.as_tensor(targets, dtype=dtype)
    target_bones = torch.as_tensor(target_bones_np, dtype=dtype)
    beta_raw0 = np.arctanh(np.clip(beta0 / beta_limit, -0.98, 0.98))
    orient_raw0 = np.arctanh(np.clip(orient0 / pose_limit, -0.98, 0.98))
    pose_raw0 = np.arctanh(np.clip(pose0 / pose_limit, -0.98, 0.98))
    beta = torch.tensor(beta_raw0, dtype=dtype, requires_grad=optimize_shape)
    pose_raw = torch.tensor(pose_raw0, dtype=dtype, requires_grad=True)
    orient_raw = torch.tensor(orient_raw0, dtype=dtype, requires_grad=True)
    translation = torch.tensor(translation0, dtype=dtype, requires_grad=True)
    log_scale = torch.tensor(math.log(max(scale0, 1e-8)), dtype=dtype,
                             requires_grad=optimize_shape)
    parameters = [pose_raw, orient_raw, translation]
    if optimize_shape:
        parameters.extend([beta, log_scale])
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    history = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        total, terms = _objective(
            model, target, target_bones, weights, d_ref, beta, pose_raw,
            orient_raw, translation, log_scale, beta_limit, pose_limit,
            optimize_shape, shape_prior=shape_prior, pose_prior=pose_prior,
            bone_weight=bone_weight, smooth_weight=smooth_weight,
            translation_prior=translation_prior,
        )
        total.backward()
        optimizer.step()
        if step == 0 or (step + 1) % max(1, steps // 10) == 0 or step == steps - 1:
            history.append({
                "step": step,
                "total": float(total.detach()),
                "point": float(terms["point_loss"].detach()),
                "bone": float(terms["bone_loss"].detach()),
                "smooth": float(terms["smooth_loss"].detach()),
            })
    with torch.no_grad():
        total, terms = _objective(
            model, target, target_bones, weights, d_ref, beta, pose_raw,
            orient_raw, translation, log_scale, beta_limit, pose_limit,
            optimize_shape, shape_prior=shape_prior, pose_prior=pose_prior,
            bone_weight=bone_weight, smooth_weight=smooth_weight,
            translation_prior=translation_prior,
        )
    return {
        "beta": terms["beta_value"].cpu().numpy(),
        "scale": float(terms["scale_value"]),
        "pose": terms["pose_value"].cpu().numpy(),
        "global_orient": terms["orient_value"].cpu().numpy(),
        "transl": translation.detach().cpu().numpy(),
        "predicted": terms["predicted"].cpu().numpy(),
        "target_bones": target_bones_np,
        "d_ref": d_ref,
        "history": history,
        "metrics": {
            "total_loss": float(total),
            "point_loss": float(terms["point_loss"]),
            "bone_loss": float(terms["bone_loss"]),
            "shape_loss": float(terms["shape_loss"]),
            "pose_loss": float(terms["pose_loss"]),
            "smooth_loss": float(terms["smooth_loss"]),
            "translation_loss": float(terms["translation_loss"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="v20 MediaPipe capture JSON")
    parser.add_argument("--mano", type=Path, required=True,
                        help="MANO_LEFT.pkl")
    parser.add_argument("--output", type=Path, required=True,
                        help="output v21 JSON")
    parser.add_argument("--hand-side", default="Left",
                        help="physical hand label required in the capture JSON")
    parser.add_argument("--max-frames", type=int, default=64,
                        help="fit at most this many valid frames; 0 means all")
    parser.add_argument("--calibration-frames", type=int, default=24,
                        help="evenly sampled frames used for beta/scale estimation")
    parser.add_argument("--shape-steps", type=int, default=350)
    parser.add_argument("--motion-steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--beta-limit", type=float, default=3.0)
    parser.add_argument("--pose-limit", type=float, default=3.141592653589793)
    parser.add_argument("--shape-prior", type=float, default=1e-3)
    parser.add_argument("--pose-prior", type=float, default=2e-4)
    parser.add_argument("--bone-weight", type=float, default=0.35)
    parser.add_argument("--smooth-weight", type=float, default=0.08)
    parser.add_argument("--translation-prior", type=float, default=2e-3)
    args = parser.parse_args()
    if args.max_frames < 0 or args.calibration_frames <= 0:
        raise ValueError("max-frames must be >= 0 and calibration-frames must be positive")
    if args.shape_steps <= 0 or args.motion_steps <= 0:
        raise ValueError("optimization steps must be positive")
    capture = load_capture(args.input.resolve(), args.hand_side, args.max_frames)
    frames = capture["frames"]
    points = np.stack([frame["points_local_m"] for frame in frames], axis=0)
    timestamps = np.asarray([frame["timestamp_s"] for frame in frames], dtype=float)
    calibration_indices = evenly_spaced_indices(len(frames), args.calibration_frames)
    calibration_points = points[calibration_indices]
    print(json.dumps({
        "status": "v21_fit_started",
        "selected_frames": len(frames),
        "calibration_frames": len(calibration_indices),
        "skipped": capture["skipped"],
    }, ensure_ascii=False))

    shape_fit = optimize_batch(
        args.mano.resolve(), calibration_points,
        steps=args.shape_steps, learning_rate=args.learning_rate,
        beta_limit=args.beta_limit, pose_limit=args.pose_limit,
        shape_prior=args.shape_prior, pose_prior=args.pose_prior,
        bone_weight=args.bone_weight, smooth_weight=args.smooth_weight,
        translation_prior=args.translation_prior, optimize_shape=True,
    )
    motion_fit = optimize_batch(
        args.mano.resolve(), points,
        beta_init=shape_fit["beta"], scale_init=shape_fit["scale"],
        steps=args.motion_steps, learning_rate=args.learning_rate,
        beta_limit=args.beta_limit, pose_limit=args.pose_limit,
        shape_prior=args.shape_prior, pose_prior=args.pose_prior,
        bone_weight=args.bone_weight, smooth_weight=args.smooth_weight,
        translation_prior=args.translation_prior, optimize_shape=False,
    )

    output_frames = []
    for index, (frame, pose, orient, transl, predicted) in enumerate(zip(
            frames, motion_fit["pose"], motion_fit["global_orient"],
            motion_fit["transl"], motion_fit["predicted"])):
        point_error = np.linalg.norm(predicted - points[index], axis=1)
        output_frames.append({
            "frame_id": frame["frame_id"],
            "timestamp_s": frame["timestamp_s"],
            "name": f"frame_{frame['frame_id']:04d}",
            "phase": "v21_mediapipe_to_mano",
            "kind": "fitted_MANO_from_MediaPipe_world_landmarks",
            "theta_hand_pose": pose.tolist(),
            "beta": motion_fit["beta"].tolist(),
            "global_orient": orient.tolist(),
            "transl": transl.tolist(),
            "source_landmarks_m": points[index].tolist(),
            "fitted_landmarks_m": predicted.tolist(),
            "point_error_mm": (point_error * 1000.0).tolist(),
            "wrist_translation_world_m": frame["wrist_world_m"].tolist(),
            "palm_basis_world_from_local": frame["basis_world_from_local"].tolist(),
        })

    morphology = morphology_summary(calibration_points)
    point_errors = np.linalg.norm(motion_fit["predicted"] - points, axis=2) * 1000.0
    bone_errors = np.abs(
        np.linalg.norm(motion_fit["predicted"][:, BONE_EDGES[:, 1]] -
                       motion_fit["predicted"][:, BONE_EDGES[:, 0]], axis=2)
        - np.linalg.norm(points[:, BONE_EDGES[:, 1]] - points[:, BONE_EDGES[:, 0]], axis=2)
    ) * 1000.0
    output_payload = {
        "version": "v21_mediapipe_to_mano_sequence",
        "status": "completed_offline_sequence_fit",
        "source": str(args.input.resolve()),
        "mano_source": str(args.mano.resolve()),
        "hand_side": args.hand_side,
        "units": {"landmarks": "m", "theta": "rad", "timestamp": "s"},
        "coordinate_frame": {
            "type": "per_frame_wrist_centered_palm_aligned",
            "x_axis": "pinky_mcp_to_index_mcp",
            "y_axis": "wrist_to_middle_mcp",
            "z_axis": "cross(x, y), right_handed",
            "reflection": "none",
        },
        "frame_count": len(output_frames),
        "calibration_frame_ids": [frames[int(i)]["frame_id"] for i in calibration_indices],
        "skipped": capture["skipped"],
        "morphology_calibration": morphology,
        "shape_fit": {
            "beta_i": motion_fit["beta"].tolist(),
            "scale_i": float(motion_fit["scale"]),
            "shape_stage_metrics": shape_fit["metrics"],
            "shape_stage_history": shape_fit["history"],
        },
        "motion_fit": {
            "motion_stage_metrics": motion_fit["metrics"],
            "motion_stage_history": motion_fit["history"],
            "point_rmse_mm": float(np.sqrt(np.mean(point_errors ** 2))),
            "point_max_mm": float(np.max(point_errors)),
            "bone_length_mae_mm": float(np.mean(bone_errors)),
            "bone_length_max_mm": float(np.max(bone_errors)),
        },
        "shared_R_stage": {
            "status": "not_run",
            "next_script": "prepare_offline_mano_sequence_v20.py",
            "note": "This file is MANO output; it is not target R landmarks and not L20 q/u.",
        },
        "frames": output_frames,
        "limitations": [
            "MediaPipe world landmarks are fitted observations, not MANO ground truth.",
            "The per-frame palm frame removes global camera/wrist orientation; original wrist positions are retained as metadata.",
            "beta_i and scale_i are estimated from the selected calibration frames and then frozen for motion fitting.",
            "No shared-R mapping, L20 q/mimic mapping, collision/contact model or actuator u is run here.",
            "This smoke output is not a cross-operator generalization result.",
        ],
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path = output.with_name(output.stem + "_metrics.json")
    summary_path.write_text(json.dumps({
        "version": output_payload["version"],
        "status": output_payload["status"],
        "source": output_payload["source"],
        "frame_count": output_payload["frame_count"],
        "calibration_frame_ids": output_payload["calibration_frame_ids"],
        "morphology_calibration": morphology,
        "shape_fit": output_payload["shape_fit"],
        "motion_fit": output_payload["motion_fit"],
        "skipped": capture["skipped"],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": output_payload["status"],
        "frames": len(output_frames),
        "beta_norm": float(np.linalg.norm(motion_fit["beta"])),
        "scale_i": float(motion_fit["scale"]),
        "point_rmse_mm": output_payload["motion_fit"]["point_rmse_mm"],
        "bone_length_mae_mm": output_payload["motion_fit"]["bone_length_mae_mm"],
        "output": str(output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
