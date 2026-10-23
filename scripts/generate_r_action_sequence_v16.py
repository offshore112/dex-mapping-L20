"""Generate a deterministic open->pinch->opposition MANO-R sequence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mano", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames-per-segment", type=int, default=9)
    args = parser.parse_args()
    if args.frames_per_segment < 2:
        raise ValueError("frames-per-segment must be at least 2")

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
    beta = torch.tensor(arrays["beta_R"], dtype=torch.float64)
    scale = float(np.asarray(arrays["scale"]))
    rotation = np.asarray(arrays["rotation"], dtype=float)
    translation = np.asarray(arrays["translation_m"], dtype=float)
    data = read_mano(args.mano.resolve())
    mano = smplx.MANO(str(args.mano.resolve()), data_struct=Struct(**data),
                      is_rhand=False, use_pca=False, flat_hand_mean=True,
                      num_betas=10, dtype=torch.float64)

    action_payload = json.loads(args.actions.resolve().read_text(encoding="utf-8"))
    action_map = {item["name"]: item for item in action_payload["actions"]}
    required = ["open", "pinch_thumb_index", "opposition_thumb_middle"]
    missing = [name for name in required if name not in action_map]
    if missing:
        raise ValueError(f"Missing fixed actions: {missing}")
    sequence = []
    frame_id = 0
    transitions = [
        ("open", "pinch_thumb_index", "open_to_pinch"),
        ("pinch_thumb_index", "opposition_thumb_middle", "pinch_to_opposition"),
    ]
    for segment_id, (start_name, end_name, phase) in enumerate(transitions):
        start = np.asarray(action_map[start_name]["theta_hand_pose"], dtype=float)
        end = np.asarray(action_map[end_name]["theta_hand_pose"], dtype=float)
        for local_id, alpha in enumerate(np.linspace(0.0, 1.0, args.frames_per_segment)):
            if segment_id == 1 and local_id == 0:
                continue  # avoid duplicating the pinch frame
            theta_np = (1.0 - alpha) * start + alpha * end
            theta = torch.tensor(theta_np, dtype=torch.float64)
            with torch.no_grad():
                result = mano(
                    betas=beta[None], hand_pose=theta[None],
                    global_orient=torch.zeros(1, 3, dtype=torch.float64),
                    transl=torch.zeros(1, 3, dtype=torch.float64),
                )
            points = mano_landmarks_torch(result.vertices[0], result.joints[0, :16]).numpy()
            points = scale * (points @ rotation.T) + translation
            if phase == "open_to_pinch":
                tip_pair = ["thumb_tip", "index_tip"] if alpha > 0.45 else None
            else:
                tip_pair = ["thumb_tip", "middle_tip"]
            sequence.append({
                "frame_id": frame_id,
                "phase": phase,
                "segment_local_id": local_id,
                "alpha": float(alpha),
                "name": f"frame_{frame_id:03d}",
                "kind": "synthetic_MANO_R_sequence",
                "theta_hand_pose": theta_np.tolist(),
                "target_landmarks_m": points.tolist(),
                "tip_pair": tip_pair,
            })
            frame_id += 1

    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "v16_R_action_sequence",
        "status": "generated_fixed_synthetic_open_pinch_opposition_sequence",
        "source_model": str(manifest_path),
        "source_actions": str(args.actions.resolve()),
        "frame_count": len(sequence),
        "frames_per_segment": args.frames_per_segment,
        "sequence": sequence,
        "limitations": [
            "The sequence is a deterministic interpolation of three fixed MANO pose vectors.",
            "It is not a camera sequence, dataset split or human demonstration.",
        ],
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "frame_count": len(sequence), "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
