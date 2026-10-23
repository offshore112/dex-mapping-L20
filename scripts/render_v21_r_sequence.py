"""Render a v21 MANO->R sequence as PNG frames, GIF, and contact sheet.

This is a read-only visualization helper.  It does not run L20 q/mimic
optimization and does not open a GUI window.  The output directory must use an
ASCII name because the current Windows MuJoCo/renderer setup is sensitive to
non-ASCII resource paths; this script itself only uses Matplotlib and Pillow.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np


def export_frame(path: Path, vertices: np.ndarray, faces: np.ndarray,
                 points: np.ndarray, frame_id: int, limits: tuple[np.ndarray, np.ndarray]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    lo, hi = limits
    center = (lo + hi) / 2.0
    span = float(np.max(hi - lo))
    fig = plt.figure(figsize=(6.4, 6.4), dpi=110)
    ax = fig.add_subplot(111, projection="3d")
    polygons = vertices[faces]
    ax.add_collection3d(Poly3DCollection(
        polygons, facecolor="#3d87b8", edgecolor="#245a7a",
        linewidth=0.08, alpha=0.38
    ))
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               s=12, c="#d63636", depthshade=False)
    chains = [
        [0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12],
        [0, 13, 14, 15, 16], [0, 17, 18, 19, 20],
    ]
    for chain in chains:
        curve = points[chain]
        ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], color="#d63636", linewidth=1.0)
    for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
        setter(c - span * 0.55, c + span * 0.55)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("R x (m)")
    ax.set_ylabel("R y (m)")
    ax.set_zlabel("R z (m)")
    ax.view_init(elev=20, azim=-72)
    ax.set_title(f"v21 shared-R visualization | frame {frame_id}")
    fig.text(0.5, 0.02, "Visualization only; no L20 q/u control result.",
             ha="center", fontsize=8, color="#526777")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mano", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True,
                        help="shared-R manifest with arrays.npz")
    parser.add_argument("--sequence", type=Path, required=True,
                        help="v21 MANO or v20 MANO->R JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=24,
                        help="evenly sample at most this many frames; 0 means all")
    parser.add_argument("--no-contact-sheet", action="store_true",
                        help="write GIF and per-frame PNGs without the large contact sheet")
    args = parser.parse_args()
    if args.max_frames < 0:
        raise ValueError("max-frames must be >= 0")

    import torch
    import smplx
    from PIL import Image
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
    if not isinstance(frames, list) or not frames:
        raise ValueError("sequence must contain a non-empty frames list")
    if args.max_frames > 0 and len(frames) > args.max_frames:
        indices = np.unique(np.linspace(0, len(frames) - 1, args.max_frames).round().astype(int))
        frames = [frames[int(i)] for i in indices]

    rendered = []
    all_vertices = []
    records = []
    for frame in frames:
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
        all_vertices.append(vertices)
        all_vertices.append(points)
        rendered.append((int(frame.get("frame_id", len(rendered))), vertices, points))

    all_points = np.vstack(all_vertices)
    lo, hi = all_points.min(axis=0), all_points.max(axis=0)
    limits = (lo, hi)
    images = []
    for frame_id, vertices, points in rendered:
        filename = f"frame_{frame_id:04d}.png"
        path = out / filename
        export_frame(path, vertices, data["f"], points, frame_id, limits)
        images.append(Image.open(path).convert("RGB"))
        records.append({"frame_id": frame_id, "image": filename})
    images[0].save(out / "mano_r_sequence.gif", save_all=True,
                   append_images=images[1:], duration=160, loop=0)
    contact_sheet = None
    if not args.no_contact_sheet:
        columns = min(4, len(images))
        rows = int(np.ceil(len(images) / columns))
        width, height = images[0].size
        sheet = Image.new("RGB", (width * columns, height * rows), "white")
        for index, image in enumerate(images):
            sheet.paste(image, ((index % columns) * width, (index // columns) * height))
        sheet.save(out / "mano_r_contact_sheet.png")
        contact_sheet = "mano_r_contact_sheet.png"
    (out / "render_info.json").write_text(json.dumps({
        "version": "v21_shared_R_visualization",
        "status": "completed_png_gif_without_contact_sheet" if args.no_contact_sheet else "completed_png_gif_contact_sheet",
        "source_sequence": str(args.sequence.resolve()),
        "source_model_manifest": str(manifest_path),
        "frame_count": len(records),
        "records": records,
        "limitations": [
            "This renders the MANO-R surface and landmarks only.",
            "It does not render L20 q, collision/contact, or actuator u.",
            "The displayed surface uses beta from each frame when present; this follows the current v20 interface and is not yet the final shared-shape policy.",
            "Contact sheet was omitted." if args.no_contact_sheet else "Contact sheet was generated.",
        ],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "completed_png_gif_without_contact_sheet" if args.no_contact_sheet else "completed_png_gif_contact_sheet",
        "frame_count": len(records),
        "gif": str(out / "mano_r_sequence.gif"),
        "contact_sheet": contact_sheet,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
