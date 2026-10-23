"""Generate one natural MANO-R OBJ for each fixed v15 synthetic action."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def export_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    lines = ["# Dynamic v15 MANO-R surface; metres"]
    lines.extend("v " + " ".join(f"{float(x):.10f}" for x in row) for row in vertices)
    lines.extend("f " + " ".join(str(int(x) + 1) for x in row) for row in faces)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mano", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import smplx
    import torch
    from smplx.utils import Struct
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from check_mano_model import read_mano

    manifest_path = args.model_manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    arrays = np.load(manifest_path.parent / manifest["arrays"]["npz"], allow_pickle=False)
    beta = torch.tensor(arrays["beta_R"], dtype=torch.float64)
    scale = float(np.asarray(arrays["scale"]))
    rotation = np.asarray(arrays["rotation"], dtype=float)
    translation = np.asarray(arrays["translation_m"], dtype=float)
    mano_data = read_mano(args.mano.resolve())
    mano = smplx.MANO(str(args.mano.resolve()), data_struct=Struct(**mano_data),
                      is_rhand=False, use_pca=False, flat_hand_mean=True,
                      num_betas=10, dtype=torch.float64)
    actions = json.loads(args.actions.resolve().read_text(encoding="utf-8"))["actions"]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for action in actions:
        theta = torch.tensor(action["theta_hand_pose"], dtype=torch.float64)
        with torch.no_grad():
            result = mano(
                betas=beta[None], hand_pose=theta[None],
                global_orient=torch.zeros(1, 3, dtype=torch.float64),
                transl=torch.zeros(1, 3, dtype=torch.float64),
            )
        vertices = scale * (result.vertices[0].numpy() @ rotation.T) + translation
        path = out / f"mano_r_{action['name']}.obj"
        export_obj(path, vertices, mano_data["f"])
        records.append({"action": action["name"], "obj": path.name, "vertex_count": int(vertices.shape[0])})
    (out / "mesh_info.json").write_text(json.dumps({
        "version": "v15_dynamic_R_action_meshes",
        "status": "generated_dynamic_natural_MANO_R_action_meshes",
        "source_model_manifest": str(manifest_path),
        "records": records,
        "limitation": "Meshes are fixed synthetic action frames, not camera estimates.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "generated_dynamic_natural_MANO_R_action_meshes", "records": records}, ensure_ascii=False))


if __name__ == "__main__":
    main()
