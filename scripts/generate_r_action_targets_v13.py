"""Generate fixed synthetic R action targets for the v13 q-level baseline."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_mano_model import read_mano  # noqa: E402
from fit_mano_r import mano_landmarks_torch  # noqa: E402


def make_theta(updates: dict[int, float]) -> np.ndarray:
    theta = np.zeros(45, dtype=float)
    for index, value in updates.items():
        theta[index] = value
    return theta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mano", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", default="v14", help="artifact version label")
    args = parser.parse_args()

    import smplx
    import torch
    from smplx.utils import Struct

    manifest_path = args.model.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    arrays = np.load(manifest_path.parent / manifest["arrays"]["npz"], allow_pickle=False)
    beta = torch.tensor(arrays["beta_R"], dtype=torch.float64)
    scale = float(arrays["scale"])
    rotation = np.asarray(arrays["rotation"], dtype=float)
    translation = np.asarray(arrays["translation_m"], dtype=float)
    data = read_mano(args.mano.resolve())
    model = smplx.MANO(
        str(args.mano.resolve()), data_struct=Struct(**data), is_rhand=False,
        use_pca=False, flat_hand_mean=True, num_betas=10, dtype=torch.float64,
    )

    # Index/middle flexion and thumb opposition vectors were fixed once for
    # this smoke baseline. They are synthetic R targets, not camera estimates.
    action_updates = {
        "open": {},
        "pinch_thumb_index": {
            3: -0.0373851720, 4: -0.7447062014, 5: -0.9683240153,
            6: -0.0132871236, 7: -0.6846395576, 8: -0.9356525520,
            9: -0.0001790803, 10: 0.0000536143, 11: -0.0004065489,
            36: 0.0229231666, 37: 0.3588915891, 38: -0.1446394554,
            39: -0.0695451490, 40: 0.1586472123, 41: 0.0145456041,
            42: 0.0806339149, 43: 0.0960213634, 44: 0.1072295820,
        },
        "opposition_thumb_middle": {
            12: -0.3985947187, 13: -0.9094331694, 14: -1.0554149050,
            15: 0.2417102136, 16: -0.9253693561, 17: -0.7884663791,
            18: 0.0000148476, 19: -0.0000902168, 20: -0.0001599150,
            36: -0.2361507740, 37: 0.5285350842, 38: -0.1449491463,
            39: -0.1353357119, 40: 0.2472566243, 41: 0.0639591355,
            42: 0.1388762565, 43: 0.1201116168, 44: 0.1785675203,
        },
    }
    tip_pairs = {
        "open": None,
        "pinch_thumb_index": ["thumb_tip", "index_tip"],
        "opposition_thumb_middle": ["thumb_tip", "middle_tip"],
    }
    point_order = manifest["kinematics"]["point_order"]
    actions = []
    for name, updates in action_updates.items():
        theta = make_theta(updates)
        with torch.no_grad():
            result = model(
                betas=beta[None], hand_pose=torch.tensor(theta)[None],
                global_orient=torch.zeros(1, 3, dtype=torch.float64),
                transl=torch.zeros(1, 3, dtype=torch.float64),
            )
        points = mano_landmarks_torch(result.vertices[0], result.joints[0, :16]).numpy()
        points = scale * (points @ rotation.T) + translation
        action = {
            "name": name,
            "kind": "synthetic_R_target",
            "theta_hand_pose": theta.tolist(),
            "target_landmarks_m": points.tolist(),
            "tip_pair": tip_pairs[name],
            "note": "Synthetic fixed target for q-level baseline; not a camera estimate.",
        }
        if tip_pairs[name] is not None:
            i, j = point_order.index(tip_pairs[name][0]), point_order.index(tip_pairs[name][1])
            action["target_tip_distance_mm"] = float(np.linalg.norm(points[i] - points[j]) * 1000)
        actions.append(action)
    payload = {
        "version": f"{args.version}_R_action_targets",
        "status": "generated_fixed_synthetic_targets",
        "source_model": str(manifest_path),
        "point_order": point_order,
        "actions": actions,
        "limitations": [
            "Targets are generated from explicit MANO pose vectors, not camera data.",
            "The pinch/opposition pose vectors are diagnostic smoke actions, not a dataset.",
        ],
    }
    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "actions": [a["name"] for a in actions]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
