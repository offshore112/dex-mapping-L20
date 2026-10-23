"""Fit a simulation-only MANO-R geometry to the 21-point L20 target.

This run keeps the MANO reference pose fixed and fits only beta, a global
positive scale, and a rigid alignment. Robot mimic/actuator constraints are
intentionally outside this geometry fit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_mano_model import (  # noqa: E402
    TIP_VERTEX_IDS,
    canonical_landmarks,
    export_obj,
    read_mano,
)


def rotation_matrix_to_axis_angle(rotation: np.ndarray) -> np.ndarray:
    value = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(value))
    if angle < 1e-8:
        return 0.5 * np.array([
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ])
    axis = np.array([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ]) / (2.0 * np.sin(angle))
    return axis * angle


def weighted_similarity(source: np.ndarray, target: np.ndarray, weights: np.ndarray):
    w = weights / weights.sum()
    source_center = (source * w[:, None]).sum(axis=0)
    target_center = (target * w[:, None]).sum(axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    covariance = (source_zero * w[:, None]).T @ target_zero
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    rotated = source_zero @ rotation.T
    numerator = float((w[:, None] * rotated * target_zero).sum())
    denominator = float((w[:, None] * source_zero * source_zero).sum())
    scale = numerator / denominator
    translation = target_center - scale * (rotation @ source_center)
    return scale, rotation, translation


def torch_axis_angle_matrix(vector):
    import torch

    theta2 = (vector * vector).sum()
    theta = torch.sqrt(theta2 + 1e-12)
    x, y, z = vector[0], vector[1], vector[2]
    skew = torch.stack((
        torch.stack((torch.zeros_like(x), -z, y)),
        torch.stack((z, torch.zeros_like(x), -x)),
        torch.stack((-y, x, torch.zeros_like(x))),
    ))
    safe = theta2 > 1e-10
    a_regular = torch.sin(theta) / theta
    b_regular = (1.0 - torch.cos(theta)) / theta2
    a_small = 1.0 - theta2 / 6.0
    b_small = 0.5 - theta2 / 24.0
    a = torch.where(safe, a_regular, a_small)
    b = torch.where(safe, b_regular, b_small)
    return torch.eye(3, dtype=vector.dtype, device=vector.device) + a * skew + b * (skew @ skew)


def mano_landmarks_torch(vertices, joints):
    """Differentiable equivalent of check_mano_model.canonical_landmarks."""
    import torch

    ids = [744, 320, 443, 554, 671]
    points = [joints[0]]
    points.extend([joints[13], joints[14], joints[15], vertices[ids[0]]])
    points.extend([joints[1], joints[2], joints[3], vertices[ids[1]]])
    points.extend([joints[4], joints[5], joints[6], vertices[ids[2]]])
    points.extend([joints[10], joints[11], joints[12], vertices[ids[3]]])
    points.extend([joints[7], joints[8], joints[9], vertices[ids[4]]])
    return torch.stack(points, dim=0)


def fit_once(model, target, weights, *, source_points, source_vertices, beta_enabled,
             pose_enabled, steps, learning_rate, beta_regularization,
             pose_regularization, dtype):
    import torch

    scale0, rotation0, translation0 = weighted_similarity(source_points, target, weights)
    log_scale = torch.tensor(math.log(scale0), dtype=dtype, requires_grad=True)
    rotvec = torch.tensor(rotation_matrix_to_axis_angle(rotation0), dtype=dtype, requires_grad=True)
    translation = torch.tensor(translation0, dtype=dtype, requires_grad=True)
    beta = torch.zeros(10, dtype=dtype, requires_grad=beta_enabled)
    hand_pose = torch.zeros(45, dtype=dtype, requires_grad=pose_enabled)
    parameters = [log_scale, rotvec, translation] + ([beta] if beta_enabled else []) + ([hand_pose] if pose_enabled else [])
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    target_t = torch.tensor(target, dtype=dtype)
    weights_t = torch.tensor(weights, dtype=dtype)
    d_ref = max(float(np.ptp(target, axis=0).max()), 1e-3)
    best = None

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        result = model(
            betas=beta.unsqueeze(0),
            hand_pose=hand_pose.unsqueeze(0),
            global_orient=torch.zeros(1, 3, dtype=dtype),
            transl=torch.zeros(1, 3, dtype=dtype),
        )
        mano_points = mano_landmarks_torch(result.vertices[0], result.joints[0, :16])
        rotation = torch_axis_angle_matrix(rotvec)
        prediction = torch.exp(log_scale) * (mano_points @ rotation.T) + translation
        residual = prediction - target_t
        point_loss = (weights_t[:, None] * residual.square()).sum() / weights_t.sum() / (d_ref * d_ref)
        beta_loss = beta_regularization * beta.square().mean() if beta_enabled else torch.zeros_like(point_loss)
        pose_loss = pose_regularization * hand_pose.square().mean() if pose_enabled else torch.zeros_like(point_loss)
        loss = point_loss + beta_loss + pose_loss
        loss.backward()
        optimizer.step()
        record = {
            "step": step,
            "loss": float(loss.detach()),
            "point_loss": float(point_loss.detach()),
            "beta_loss": float(beta_loss.detach()),
            "pose_loss": float(pose_loss.detach()),
        }
        if best is None or record["loss"] < best["loss"]:
            best = record

    # A short quasi-Newton finish improves the final rigid alignment after Adam.
    optimizer = torch.optim.LBFGS(parameters, max_iter=120, history_size=20,
                                  line_search_fn="strong_wolfe", tolerance_grad=1e-10,
                                  tolerance_change=1e-12)

    def closure():
        optimizer.zero_grad(set_to_none=True)
        result = model(
            betas=beta.unsqueeze(0),
            hand_pose=hand_pose.unsqueeze(0),
            global_orient=torch.zeros(1, 3, dtype=dtype),
            transl=torch.zeros(1, 3, dtype=dtype),
        )
        mano_points = mano_landmarks_torch(result.vertices[0], result.joints[0, :16])
        rotation = torch_axis_angle_matrix(rotvec)
        prediction = torch.exp(log_scale) * (mano_points @ rotation.T) + translation
        residual = prediction - target_t
        point_loss = (weights_t[:, None] * residual.square()).sum() / weights_t.sum() / (d_ref * d_ref)
        beta_loss = beta_regularization * beta.square().mean() if beta_enabled else torch.zeros_like(point_loss)
        pose_loss = pose_regularization * hand_pose.square().mean() if pose_enabled else torch.zeros_like(point_loss)
        total = point_loss + beta_loss + pose_loss
        total.backward()
        return total

    optimizer.step(closure)
    with torch.no_grad():
        result = model(
            betas=beta.unsqueeze(0),
            hand_pose=hand_pose.unsqueeze(0),
            global_orient=torch.zeros(1, 3, dtype=dtype),
            transl=torch.zeros(1, 3, dtype=dtype),
        )
        mano_points = mano_landmarks_torch(result.vertices[0], result.joints[0, :16])
        rotation = torch_axis_angle_matrix(rotvec)
        fitted_points = torch.exp(log_scale) * (mano_points @ rotation.T) + translation
        fitted_vertices = torch.exp(log_scale) * (result.vertices[0] @ rotation.T) + translation
        residual = fitted_points - target_t
        point_error = torch.linalg.norm(residual, dim=1).cpu().numpy()
        point_loss = float((weights_t[:, None] * residual.square()).sum() / weights_t.sum() / (d_ref * d_ref))
        beta_loss = float(beta_regularization * beta.square().mean()) if beta_enabled else 0.0
        pose_loss = float(pose_regularization * hand_pose.square().mean()) if pose_enabled else 0.0
        total_loss = point_loss + beta_loss + pose_loss
        state = {
            "beta": beta.cpu().numpy().copy(),
            "hand_pose": hand_pose.cpu().numpy().copy(),
            "scale": float(torch.exp(log_scale)),
            "rotation": rotation.cpu().numpy().copy(),
            "rotation_axis_angle": rotvec.cpu().numpy().copy(),
            "translation": translation.cpu().numpy().copy(),
            "fitted_points": fitted_points.cpu().numpy().copy(),
            "fitted_vertices": fitted_vertices.cpu().numpy().copy(),
            "point_error_m": point_error,
            "point_loss": point_loss,
            "beta_regularization_loss": beta_loss,
            "pose_regularization_loss": pose_loss,
            "total_loss": total_loss,
            "optimizer_best_adam": best,
            "steps_adam": steps,
            "lbfgs_max_iter": 120,
            "beta_enabled": beta_enabled,
            "pose_enabled": pose_enabled,
        }
    return state


def group_metrics(errors_m, names):
    groups = {"wrist": [0], "thumb": [1, 2, 3, 4], "index": [5, 6, 7, 8],
              "middle": [9, 10, 11, 12], "ring": [13, 14, 15, 16],
              "pinky": [17, 18, 19, 20]}
    out = {}
    for group, ids in groups.items():
        values = errors_m[ids] * 1000.0
        out[group] = {"rmse_mm": float(np.sqrt(np.mean(values ** 2))),
                      "mean_mm": float(np.mean(values)), "max_mm": float(np.max(values)),
                      "point_names": [names[i] for i in ids]}
    return out


def render_overlay(path, vertices, faces, target, fitted, names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig, ax = plt.subplots(figsize=(10, 9), subplot_kw={"projection": "3d"})
    polygons = vertices[faces] * 1000.0
    ax.add_collection3d(Poly3DCollection(polygons, facecolor="#3c83ad", edgecolor="#265873",
                                         linewidth=0.05, alpha=0.24, label="MANO-R surface"))
    colors = {"thumb": "#d98243", "index": "#3b82ac", "middle": "#569678",
              "ring": "#946cb0", "pinky": "#c76c8a"}
    chains = {"thumb": [0, 1, 2, 3, 4], "index": [0, 5, 6, 7, 8],
              "middle": [0, 9, 10, 11, 12], "ring": [0, 13, 14, 15, 16],
              "pinky": [0, 17, 18, 19, 20]}
    target_mm = target * 1000.0
    fitted_mm = fitted * 1000.0
    ax.scatter(target_mm[:, 0], target_mm[:, 1], target_mm[:, 2], c="#d23c3c", s=24,
               label="L20 target points")
    ax.scatter(fitted_mm[:, 0], fitted_mm[:, 1], fitted_mm[:, 2], c="#2b9b62", s=18,
               label="fitted MANO points")
    for finger, ids in chains.items():
        c = colors[finger]
        ax.plot(target_mm[ids, 0], target_mm[ids, 1], target_mm[ids, 2], color=c,
                linewidth=1.7, alpha=0.8)
        ax.plot(fitted_mm[ids, 0], fitted_mm[ids, 1], fitted_mm[ids, 2], color=c,
                linewidth=1.2, linestyle="--", alpha=0.9)
    all_points = np.vstack((vertices, target, fitted)) * 1000.0
    lo, hi = all_points.min(axis=0), all_points.max(axis=0)
    center, span = (lo + hi) / 2.0, float(np.max(hi - lo))
    for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
        setter(c - span * 0.56, c + span * 0.56)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("root X (mm)"); ax.set_ylabel("root Y (mm)"); ax.set_zlabel("root Z (mm)")
    ax.view_init(elev=20, azim=-72)
    ax.set_title("MANO-R fitted to L20 zero-pose geometry\nred=L20 target, green=fitted MANO points")
    ax.legend(loc="upper left", fontsize=8)
    fig.text(0.5, 0.02, "Simulation/CAD fit only; no hardware calibration or actuator mapping.",
             ha="center", fontsize=9, color="#526777")
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mano", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1600)
    args = parser.parse_args()

    import torch
    import smplx
    from smplx.utils import Struct

    torch.set_num_threads(1)
    dtype = torch.float64
    mano_data = read_mano(args.mano.resolve())
    model = smplx.MANO(str(args.mano.resolve()), data_struct=Struct(**mano_data),
                       is_rhand=False, use_pca=False, flat_hand_mean=True,
                       num_betas=10, dtype=dtype)
    target_record = json.loads(args.target.read_text(encoding="utf-8"))
    names = target_record["point_order"]
    target = np.asarray([p["xyz_m"] for p in target_record["points"]], dtype=float)
    if target.shape != (21, 3):
        raise ValueError(f"Expected a 21x3 target, got {target.shape}")
    weights = np.ones(21, dtype=float)
    weights[0] = 2.0

    with torch.no_grad():
        neutral = model(
            betas=torch.zeros(1, 10, dtype=dtype), hand_pose=torch.zeros(1, 45, dtype=dtype),
            global_orient=torch.zeros(1, 3, dtype=dtype), transl=torch.zeros(1, 3, dtype=dtype),
        )
        neutral_points = mano_landmarks_torch(neutral.vertices[0], neutral.joints[0, :16]).cpu().numpy()

    baseline = fit_once(model, target, weights, source_points=neutral_points,
                        source_vertices=neutral.vertices[0].detach().cpu().numpy(),
                        beta_enabled=False, pose_enabled=False,
                        steps=max(250, args.steps // 3), learning_rate=0.03,
                        beta_regularization=0.0, pose_regularization=0.0, dtype=dtype)
    shaped = fit_once(model, target, weights, source_points=neutral_points,
                      source_vertices=neutral.vertices[0].detach().cpu().numpy(),
                      beta_enabled=True, pose_enabled=True, steps=args.steps, learning_rate=0.02,
                      beta_regularization=1e-4, pose_regularization=1e-4, dtype=dtype)

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "fit_summary.json").exists():
        raise FileExistsError(f"Preserve previous run: choose a new output directory: {out}")

    result = {
        "version": "v5_mano_r_fit",
        "status": "completed_simulation_geometry_fit",
        "fit_scope": "fitted fixed MANO reference pose theta_ref; beta + positive scale + rigid alignment",
        "hardware": "not used; no hardware calibration",
        "mano_source": str(args.mano.resolve()),
        "mano_sha256": hashlib.sha256(args.mano.resolve().read_bytes()).hexdigest(),
        "target_source": str(args.target.resolve()),
        "landmark_order": names,
        "tip_vertex_ids": TIP_VERTEX_IDS,
        "weights": weights.tolist(),
        "baseline_scale_only": {
            "scale": baseline["scale"], "rotation": baseline["rotation"].tolist(),
            "translation_m": baseline["translation"].tolist(),
            "point_error_mm": (baseline["point_error_m"] * 1000).tolist(),
            "rmse_mm": float(np.sqrt(np.mean((baseline["point_error_m"] * 1000) ** 2))),
            "max_mm": float(np.max(baseline["point_error_m"] * 1000)),
            "group_metrics": group_metrics(baseline["point_error_m"], names),
            "total_loss": baseline["total_loss"],
        },
        "beta_scale_fit": {
            "beta": shaped["beta"].tolist(), "theta_ref_hand_pose": shaped["hand_pose"].tolist(),
            "scale": shaped["scale"],
            "rotation": shaped["rotation"].tolist(),
            "rotation_axis_angle": shaped["rotation_axis_angle"].tolist(),
            "translation_m": shaped["translation"].tolist(),
            "point_error_mm": (shaped["point_error_m"] * 1000).tolist(),
            "rmse_mm": float(np.sqrt(np.mean((shaped["point_error_m"] * 1000) ** 2))),
            "mean_mm": float(np.mean(shaped["point_error_m"] * 1000)),
            "max_mm": float(np.max(shaped["point_error_m"] * 1000)),
            "group_metrics": group_metrics(shaped["point_error_m"], names),
            "point_loss": shaped["point_loss"],
            "beta_regularization_loss": shaped["beta_regularization_loss"],
            "pose_regularization_loss": shaped["pose_regularization_loss"],
            "total_loss": shaped["total_loss"],
            "optimizer": {"adam_steps": args.steps, "lbfgs_max_iter": 120,
                          "learning_rate": 0.02, "beta_regularization": 1e-4,
                          "pose_enabled": True, "pose_regularization": 1e-4},
        },
        "limitations": [
            "L20 joint origins are mechanical proxies for MANO joints.",
            "The fit uses one URDF zero-pose geometry target; it is not a dynamic retargeting result.",
            "MANO-R does not include L20 mimic or actuator constraints; those belong to stage two.",
            "Tip points are distal-cap centers, not measured contact centers.",
        ],
    }
    (out / "fit_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    np.savez(out / "mano_r_fit.npz", beta=shaped["beta"], theta_ref_hand_pose=shaped["hand_pose"],
             scale=shaped["scale"],
             rotation=shaped["rotation"], translation_m=shaped["translation"],
             vertices_m=shaped["fitted_vertices"], fitted_points_m=shaped["fitted_points"],
             target_points_m=target, faces=mano_data["f"], point_error_m=shaped["point_error_m"])
    export_obj(out / "mano_r_fitted.obj", shaped["fitted_vertices"], mano_data["f"])
    export_obj(out / "mano_neutral_aligned.obj", baseline["fitted_vertices"], mano_data["f"])
    render_overlay(out / "mano_r_l20_overlay.png", shaped["fitted_vertices"], mano_data["f"],
                   target, shaped["fitted_points"], names)
    (out / "fitted_landmarks.json").write_text(json.dumps({
        "unit": "m", "frame": "hand_base_link", "landmark_order": names,
        "target_points_m": target.tolist(), "fitted_points_m": shaped["fitted_points"].tolist(),
        "point_error_mm": (shaped["point_error_m"] * 1000).tolist(),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(out),
                      "baseline_rmse_mm": result["baseline_scale_only"]["rmse_mm"],
                      "beta_rmse_mm": result["beta_scale_fit"]["rmse_mm"],
                      "beta_max_mm": result["beta_scale_fit"]["max_mm"],
                      "scale": shaped["scale"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
