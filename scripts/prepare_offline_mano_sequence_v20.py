"""Normalize an offline camera/MANO JSON sequence for R->L20 evaluation.

The accepted input is a JSON object with either ``frames`` or ``sequence``.
Each frame must contain a 45-value ``theta_hand_pose``.  Optional MANO fields
are ``beta`` (10), ``global_orient`` (3), ``transl`` (3), ``timestamp_s`` and
``tip_pair``.  The current interface uses a fixed MANO->R transform from the
shared model as a reproducible normalization baseline; it is not the learned
cross-operator stage yet.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--mano", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-rate-hz", type=float, default=30.0)
    parser.add_argument("--use-existing-landmarks", action="store_true",
                        help="use input target_landmarks_m instead of MANO forward for fixture replay")
    args = parser.parse_args()
    if args.frame_rate_hz <= 0:
        raise ValueError("frame-rate-hz must be positive")

    import smplx
    import torch
    from smplx.utils import Struct
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from check_mano_model import read_mano
    from fit_mano_r import mano_landmarks_torch

    manifest_path = args.model.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    arrays = np.load(manifest_path.parent / manifest["arrays"]["npz"], allow_pickle=False)
    beta_default = np.asarray(arrays["beta_R"], dtype=float)
    scale = float(np.asarray(arrays["scale"]))
    rotation = np.asarray(arrays["rotation"], dtype=float)
    translation = np.asarray(arrays["translation_m"], dtype=float)
    mano_data = read_mano(args.mano.resolve())
    mano = smplx.MANO(str(args.mano.resolve()), data_struct=Struct(**mano_data),
                      is_rhand=False, use_pca=False, flat_hand_mean=True,
                      num_betas=10, dtype=torch.float64)

    source = json.loads(args.input.resolve().read_text(encoding="utf-8"))
    raw_frames = source.get("frames", source.get("sequence"))
    if not isinstance(raw_frames, list) or not raw_frames:
        raise ValueError("Input must contain a non-empty frames or sequence list")
    output_frames = []
    for index, raw in enumerate(raw_frames):
        theta = np.asarray(raw.get("theta_hand_pose"), dtype=float)
        if theta.shape != (45,) or not np.isfinite(theta).all():
            raise ValueError(f"Frame {index} theta_hand_pose must be finite shape (45,)")
        beta = np.asarray(raw.get("beta", beta_default), dtype=float)
        if beta.shape != (10,) or not np.isfinite(beta).all():
            raise ValueError(f"Frame {index} beta must be finite shape (10,)")
        global_orient = np.asarray(raw.get("global_orient", [0.0, 0.0, 0.0]), dtype=float)
        transl = np.asarray(raw.get("transl", [0.0, 0.0, 0.0]), dtype=float)
        if global_orient.shape != (3,) or transl.shape != (3,):
            raise ValueError(f"Frame {index} global_orient/transl must have shape (3,)")
        existing = raw.get("target_landmarks_m")
        if args.use_existing_landmarks and existing is not None:
            target = np.asarray(existing, dtype=float)
        else:
            with torch.no_grad():
                result = mano(
                    betas=torch.tensor(beta, dtype=torch.float64)[None],
                    hand_pose=torch.tensor(theta, dtype=torch.float64)[None],
                    global_orient=torch.tensor(global_orient, dtype=torch.float64)[None],
                    transl=torch.tensor(transl, dtype=torch.float64)[None],
                )
            mano_points = mano_landmarks_torch(result.vertices[0], result.joints[0, :16]).numpy()
            target = scale * (mano_points @ rotation.T) + translation
        if target.shape != (21, 3) or not np.isfinite(target).all():
            raise ValueError(f"Frame {index} target landmarks must be finite shape (21,3)")
        frame_id = int(raw.get("frame_id", index))
        frame = {
            "frame_id": frame_id,
            "timestamp_s": float(raw.get("timestamp_s", index / args.frame_rate_hz)),
            "name": str(raw.get("name", f"frame_{frame_id:03d}")),
            "phase": str(raw.get("phase", "offline_mano")),
            "alpha": float(raw.get("alpha", 0.0)),
            "kind": "offline_camera_MANO_normalized_to_R",
            "theta_hand_pose": theta.tolist(),
            "beta": beta.tolist(),
            "global_orient": global_orient.tolist(),
            "transl": transl.tolist(),
            "target_landmarks_m": target.tolist(),
        }
        if raw.get("tip_pair") is not None:
            frame["tip_pair"] = raw["tip_pair"]
        output_frames.append(frame)

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "v20_offline_camera_MANO_sequence_interface",
        "status": "normalized_offline_MANO_sequence_to_R_interface",
        "source": str(args.input.resolve()),
        "model": str(manifest_path),
        "hand_side": "left",
        "units": {"landmarks": "m", "theta": "rad", "timestamp": "s"},
        "frame_rate_hz": args.frame_rate_hz,
        "frame_count": len(output_frames),
        "normalization": {
            "type": "fixed_MANO_to_R_transform_from_shared_model",
            "beta_source": "input beta when present, otherwise shared model beta_R",
            "learned_cross_operator_stage": "not_implemented",
        },
        "frames": output_frames,
        "limitations": [
            "This is an offline interface and fixed normalization baseline, not a camera detector.",
            "Camera confidence, missing frames, temporal filtering and MANO fitting uncertainty are not included.",
            "The L20 actuator command u remains undefined.",
        ],
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "frame_count": len(output_frames), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
