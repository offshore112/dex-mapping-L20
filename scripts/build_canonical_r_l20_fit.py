"""Create a canonical human-like MANO-R fit with beta=0 and flat pose.

This is intentionally a separate candidate from the beta/theta optimization:
the canonical R keeps a recognizable MANO hand shape and only uses the v5
scale-only rigid alignment to place it in the L20 root frame.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_mano_model import canonical_landmarks, export_obj, read_mano  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mano", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
    with torch.no_grad():
        output = model(
            betas=torch.zeros(1, 10, dtype=dtype),
            hand_pose=torch.zeros(1, 45, dtype=dtype),
            global_orient=torch.zeros(1, 3, dtype=dtype),
            transl=torch.zeros(1, 3, dtype=dtype),
        )
    neutral_vertices = output.vertices[0].cpu().numpy()
    neutral_points, names = canonical_landmarks(
        neutral_vertices, output.joints[0, :16].cpu().numpy()
    )
    target_record = json.loads(args.target.resolve().read_text(encoding="utf-8"))
    target = np.asarray([item["xyz_m"] for item in target_record["points"]], dtype=float)
    if len(names) != len(target_record["point_order"]):
        raise ValueError("MANO and L20 landmark counts differ")
    summary = json.loads(args.baseline_summary.resolve().read_text(encoding="utf-8"))
    baseline = summary["baseline_scale_only"]
    scale = float(baseline["scale"])
    rotation = np.asarray(baseline["rotation"], dtype=float)
    translation = np.asarray(baseline["translation_m"], dtype=float)
    vertices = scale * (neutral_vertices @ rotation.T) + translation
    fitted = scale * (neutral_points @ rotation.T) + translation
    error = np.linalg.norm(fitted - target, axis=1)

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "mano_r_fit.npz",
        beta=np.zeros(10, dtype=np.float64),
        theta_ref_hand_pose=np.zeros(45, dtype=np.float64),
        scale=np.asarray(scale, dtype=np.float64),
        rotation=rotation,
        translation_m=translation,
        vertices_m=vertices,
        fitted_points_m=fitted,
        target_points_m=target,
        faces=data["f"],
        point_error_m=error,
    )
    export_obj(out / "mano_r_fitted.obj", vertices, data["f"])
    result = {
        "version": "v12_canonical_mano_r_fit",
        "status": "constructed_canonical_beta0_flat_pose_reference",
        "fit_scope": "beta=0, flat MANO pose, scale-only rigid placement from v5 baseline",
        "mano_source": str(mano_path),
        "mano_sha256": hashlib.sha256(mano_path.read_bytes()).hexdigest(),
        "target_source": str(args.target.resolve()),
        "beta": [0.0] * 10,
        "theta_ref_hand_pose": [0.0] * 45,
        "scale": scale,
        "rotation": rotation.tolist(),
        "translation_m": translation.tolist(),
        "point_error_mm": (error * 1000).tolist(),
        "rmse_mm": float(np.sqrt(np.mean((error * 1000) ** 2))),
        "max_mm": float(np.max(error * 1000)),
        "limitations": [
            "This keeps a canonical recognizable MANO shape rather than minimizing all L20 point residuals.",
            "The residual is a static placement diagnostic, not dynamic retargeting performance.",
            "L20 q/mimic and actuator u remain separate interfaces.",
        ],
    }
    (out / "fit_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "rmse_mm": result["rmse_mm"], "max_mm": result["max_mm"], "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
