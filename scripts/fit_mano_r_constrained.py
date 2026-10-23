"""Fit a target-specific MANO-R candidate while preserving a natural pose/shape.

The hand pose is fixed to the flat MANO reference. Shape coefficients are
bounded by ``beta_max`` and regularized, so a low landmark loss cannot create
an extreme, non-anthropomorphic MANO shape. This is a candidate generator,
not a proof that the candidate is the final R_L20 representation.
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
from check_mano_model import canonical_landmarks, export_obj, read_mano  # noqa: E402
from fit_mano_r import (  # noqa: E402
    mano_landmarks_torch,
    rotation_matrix_to_axis_angle,
    torch_axis_angle_matrix,
    weighted_similarity,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mano", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--beta-max", type=float, default=3.0)
    parser.add_argument("--beta-reg", type=float, default=0.01)
    args = parser.parse_args()

    import smplx
    import torch
    from smplx.utils import Struct

    torch.set_num_threads(1)
    dtype = torch.float64
    mano_path = args.mano.resolve()
    data = read_mano(mano_path)
    model = smplx.MANO(
        str(mano_path), data_struct=Struct(**data), is_rhand=False,
        use_pca=False, flat_hand_mean=True, num_betas=10, dtype=dtype,
    )
    target_record = json.loads(args.target.resolve().read_text(encoding="utf-8"))
    target = np.asarray([item["xyz_m"] for item in target_record["points"]], dtype=float)
    weights = np.ones(len(target), dtype=float)
    weights[0] = 2.0
    with torch.no_grad():
        neutral = model(
            betas=torch.zeros(1, 10, dtype=dtype), hand_pose=torch.zeros(1, 45, dtype=dtype),
            global_orient=torch.zeros(1, 3, dtype=dtype), transl=torch.zeros(1, 3, dtype=dtype),
        )
    neutral_vertices = neutral.vertices[0].cpu().numpy()
    neutral_points, _ = canonical_landmarks(neutral_vertices, neutral.joints[0, :16].cpu().numpy())
    scale0, rotation0, translation0 = weighted_similarity(neutral_points, target, weights)
    log_scale = torch.tensor(math.log(scale0), dtype=dtype, requires_grad=True)
    rotvec = torch.tensor(rotation_matrix_to_axis_angle(rotation0), dtype=dtype, requires_grad=True)
    translation = torch.tensor(translation0, dtype=dtype, requires_grad=True)
    beta_raw = torch.zeros(10, dtype=dtype, requires_grad=True)
    beta_max = float(args.beta_max)
    beta_reg = float(args.beta_reg)
    target_t = torch.tensor(target, dtype=dtype)
    weights_t = torch.tensor(weights, dtype=dtype)
    d_ref = max(float(np.ptp(target, axis=0).max()), 1e-3)

    def evaluate():
        beta = beta_max * torch.tanh(beta_raw)
        result = model(
            betas=beta.unsqueeze(0), hand_pose=torch.zeros(1, 45, dtype=dtype),
            global_orient=torch.zeros(1, 3, dtype=dtype), transl=torch.zeros(1, 3, dtype=dtype),
        )
        points = mano_landmarks_torch(result.vertices[0], result.joints[0, :16])
        rotation = torch_axis_angle_matrix(rotvec)
        fitted = torch.exp(log_scale) * (points @ rotation.T) + translation
        vertices = torch.exp(log_scale) * (result.vertices[0] @ rotation.T) + translation
        residual = fitted - target_t
        point_loss = (weights_t[:, None] * residual.square()).sum() / weights_t.sum() / (d_ref * d_ref)
        regularization = beta_reg * (beta * beta).mean()
        return beta, fitted, vertices, point_loss + regularization, point_loss, regularization

    optimizer = torch.optim.Adam([log_scale, rotvec, translation, beta_raw], lr=0.02)
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        _, _, _, loss, _, _ = evaluate()
        loss.backward()
        optimizer.step()
    optimizer = torch.optim.LBFGS(
        [log_scale, rotvec, translation, beta_raw], max_iter=120, history_size=20,
        line_search_fn="strong_wolfe", tolerance_grad=1e-10, tolerance_change=1e-12,
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)
        _, _, _, loss, _, _ = evaluate()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        beta, fitted_t, vertices_t, total, point_loss, reg_loss = evaluate()
    beta_np = beta.cpu().numpy()
    fitted = fitted_t.cpu().numpy()
    vertices = vertices_t.cpu().numpy()
    error = np.linalg.norm(fitted - target, axis=1)
    scale = float(torch.exp(log_scale).detach())
    rotation = torch_axis_angle_matrix(rotvec).detach().cpu().numpy()
    translation_np = translation.detach().cpu().numpy()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "mano_r_fit.npz", beta=beta_np, theta_ref_hand_pose=np.zeros(45),
        scale=np.asarray(scale), rotation=rotation, translation_m=translation_np,
        vertices_m=vertices, fitted_points_m=fitted, target_points_m=target,
        faces=data["f"], point_error_m=error,
    )
    export_obj(out / "mano_r_fitted.obj", vertices, data["f"])
    summary = {
        "version": "v13_constrained_target_mano_r_fit",
        "status": "constructed_bounded_beta_flat_pose_candidate",
        "fit_scope": "flat MANO pose; beta bounded by tanh(beta_max), beta regularized; scale + rigid alignment",
        "mano_source": str(mano_path),
        "mano_sha256": hashlib.sha256(mano_path.read_bytes()).hexdigest(),
        "target_source": str(args.target.resolve()),
        "beta": beta_np.tolist(),
        "beta_max": beta_max,
        "beta_regularization": beta_reg,
        "theta_ref_hand_pose": [0.0] * 45,
        "scale": scale,
        "rotation": rotation.tolist(),
        "translation_m": translation_np.tolist(),
        "point_error_mm": (error * 1000).tolist(),
        "rmse_mm": float(np.sqrt(np.mean((error * 1000) ** 2))),
        "max_mm": float(np.max(error * 1000)),
        "beta_l2": float(np.linalg.norm(beta_np)),
        "beta_abs_max": float(np.max(np.abs(beta_np))),
        "point_loss": float(point_loss),
        "beta_regularization_loss": float(reg_loss),
        "total_loss": float(total),
        "limitations": [
            "This candidate preserves flat MANO pose and bounds beta; it is not a dense CAD surface fit.",
            "L20 q/mimic and actuator u remain separate interfaces.",
            "Natural appearance and target correspondence still require visual and task-level validation.",
        ],
    }
    (out / "fit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "rmse_mm": summary["rmse_mm"], "max_mm": summary["max_mm"], "beta_abs_max": summary["beta_abs_max"], "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
