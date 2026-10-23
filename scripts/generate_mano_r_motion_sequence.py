"""Generate a small MANO-R pose sequence from fixed v5 beta and theta_ref."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_mano_model import read_mano  # noqa: E402


def mano_landmarks_torch(vertices, joints):
    import torch
    ids = [744, 320, 443, 554, 671]
    return torch.stack([
        joints[0], joints[13], joints[14], joints[15], vertices[ids[0]],
        joints[1], joints[2], joints[3], vertices[ids[1]],
        joints[4], joints[5], joints[6], vertices[ids[2]],
        joints[10], joints[11], joints[12], vertices[ids[3]],
        joints[7], joints[8], joints[9], vertices[ids[4]],
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mano", type=Path, required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=9)
    args = parser.parse_args()

    import torch
    import smplx
    from smplx.utils import Struct

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from check_mano_model import read_mano
    data = read_mano(args.mano.resolve())
    model = smplx.MANO(str(args.mano.resolve()), data_struct=Struct(**data),
                       is_rhand=False, use_pca=False, flat_hand_mean=True,
                       num_betas=10, dtype=torch.float64)
    fit = np.load(args.fit.resolve())
    beta = torch.tensor(fit["beta"], dtype=torch.float64)
    theta_ref = torch.tensor(fit["theta_ref_hand_pose"], dtype=torch.float64)
    scale = float(fit["scale"])
    rotation = np.asarray(fit["rotation"], dtype=float)
    translation = np.asarray(fit["translation_m"], dtype=float)
    values = np.linspace(0.0, 0.45, args.frames)
    vertices = []
    for value in values:
        theta = theta_ref.clone()
        # MANO native hand-pose channels 3:6 are the index middle joint
        # (the channel is used only for a visual motion sanity check).
        theta[3:6] += torch.tensor([float(value), 0.0, 0.0], dtype=torch.float64)
        with torch.no_grad():
            result = model(
                betas=beta[None], hand_pose=theta[None],
                global_orient=torch.zeros(1, 3, dtype=torch.float64),
                transl=torch.zeros(1, 3, dtype=torch.float64),
            )
        v = result.vertices[0].numpy()
        v = scale * (v @ rotation.T) + translation
        vertices.append(v)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "mano_r_sequence.npz", vertices_m=np.asarray(vertices), channel_values=values,
             pose_channel="index_middle_axis_x_channels_3_6", beta=fit["beta"],
             theta_ref_hand_pose=theta_ref, scale=scale, rotation=rotation,
             translation_m=translation)
    (out / "sequence_info.json").write_text(json.dumps({
        "version": "v7_mano_r_motion_preview",
        "status": "generated_for_visual_motion_check",
        "frame_count": args.frames,
        "pose_channel": "MANO native hand_pose[3:6], axis-angle x increment",
        "increment_range_rad": [float(values.min()), float(values.max())],
        "fixed_beta": fit["beta"].tolist(),
        "limitation": "Not a camera estimate or L20 retargeting result.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out / 'mano_r_sequence.npz'} frames={args.frames}")


if __name__ == "__main__":
    main()
