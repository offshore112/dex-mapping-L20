"""Render selected v21 R frames from four fixed camera views."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mano", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="*", default=[0, 190, 383, 773])
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import smplx
    import torch
    from smplx.utils import Struct

    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))
    from check_mano_model import read_mano
    from fit_mano_r import mano_landmarks_torch

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(ord(char) > 127 for char in str(out)):
        raise ValueError("Use an ASCII output directory name for rendering")

    manifest_path = args.model_manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    arrays = np.load(manifest_path.parent / manifest["arrays"]["npz"], allow_pickle=False)
    beta_default = np.asarray(arrays["beta_R"], dtype=float)
    scale = float(np.asarray(arrays["scale"]))
    rotation = np.asarray(arrays["rotation"], dtype=float)
    translation = np.asarray(arrays["translation_m"], dtype=float)
    data = read_mano(args.mano.resolve())
    mano = smplx.MANO(str(args.mano.resolve()), data_struct=Struct(**data),
                      is_rhand=False, use_pca=False, flat_hand_mean=True,
                      num_betas=10, dtype=torch.float64)
    payload = json.loads(args.sequence.resolve().read_text(encoding="utf-8"))
    frames = payload.get("frames", payload.get("sequence"))
    by_id = {int(frame.get("frame_id", index)): frame for index, frame in enumerate(frames)}
    selected = [frame_id for frame_id in args.frames if frame_id in by_id]
    if not selected:
        raise ValueError(f"None of the requested frames exist: {args.frames}")

    records = []
    all_points = []
    for frame_id in selected:
        frame = by_id[frame_id]
        beta = np.asarray(frame.get("beta", beta_default), dtype=float)
        theta = np.asarray(frame["theta_hand_pose"], dtype=float)
        orient = np.asarray(frame.get("global_orient", [0.0, 0.0, 0.0]), dtype=float)
        transl = np.asarray(frame.get("transl", [0.0, 0.0, 0.0]), dtype=float)
        with torch.no_grad():
            result = mano(
                betas=torch.tensor(beta, dtype=torch.float64)[None],
                hand_pose=torch.tensor(theta, dtype=torch.float64)[None],
                global_orient=torch.tensor(orient, dtype=torch.float64)[None],
                transl=torch.tensor(transl, dtype=torch.float64)[None],
            )
        vertices = scale * (result.vertices[0].numpy() @ rotation.T) + translation
        points = mano_landmarks_torch(result.vertices[0], result.joints[0, :16]).numpy()
        points = scale * (points @ rotation.T) + translation
        tip_index = float(np.linalg.norm(points[4] - points[8]) * 1000.0)
        tip_middle = float(np.linalg.norm(points[4] - points[12]) * 1000.0)
        records.append({"frame_id": frame_id, "vertices": vertices, "points": points,
                        "thumb_index_tip_distance_mm": tip_index,
                        "thumb_middle_tip_distance_mm": tip_middle})
        all_points.extend([vertices, points])

    all_points = np.vstack(all_points)
    lo, hi = all_points.min(axis=0), all_points.max(axis=0)
    center = (lo + hi) / 2.0
    span = float(np.max(hi - lo))
    views = [("oblique", -72, 20), ("front", -90, 0), ("side", 0, 0), ("top", -90, 90)]
    chains = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12],
              [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]
    images = []
    for record in records:
        fig = plt.figure(figsize=(12, 10), dpi=110)
        for panel, (name, azimuth, elevation) in enumerate(views, start=1):
            ax = fig.add_subplot(2, 2, panel, projection="3d")
            ax.add_collection3d(Poly3DCollection(
                record["vertices"][data["f"]], facecolor="#3d87b8",
                edgecolor="#245a7a", linewidth=0.08, alpha=0.38
            ))
            pts = record["points"]
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=15, c="#d63636", depthshade=False)
            for chain in chains:
                curve = pts[chain]
                ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], color="#d63636", linewidth=1.0)
            for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
                setter(c - span * 0.55, c + span * 0.55)
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=elevation, azim=azimuth)
            ax.set_title(name)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
        fig.suptitle(
            f"v21 R frame {record['frame_id']} | thumb-index {record['thumb_index_tip_distance_mm']:.1f} mm | "
            f"thumb-middle {record['thumb_middle_tip_distance_mm']:.1f} mm",
            fontsize=13,
        )
        fig.text(0.5, 0.015, "R surface and 21-point skeleton; no L20 q/u or contact result.",
                 ha="center", fontsize=9, color="#526777")
        path = out / f"frame_{record['frame_id']:04d}_multiview.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        images.append(path)

    (out / "multiview_info.json").write_text(json.dumps({
        "version": "v21_R_multiview_diagnostic",
        "status": "completed_selected_frame_multiview",
        "frames": [{k: v for k, v in record.items() if k != "vertices" and k != "points"}
                   for record in records],
        "views": [name for name, _, _ in views],
        "limitations": [
            "This view diagnoses MANO-R shape and point geometry only.",
            "Tip distances are geometric distances; they are not contact success metrics.",
            "The displayed surface follows the current v20 per-frame beta behavior.",
        ],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed_selected_frame_multiview",
                      "frames": selected, "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
