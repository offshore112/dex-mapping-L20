"""Generate one MANO-R OBJ per frame of the v16 synthetic sequence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def export_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    lines = ["# Dynamic v16 MANO-R surface; metres"]
    lines.extend("v " + " ".join(f"{float(x):.10f}" for x in row) for row in vertices)
    lines.extend("f " + " ".join(str(int(x) + 1) for x in row) for row in faces)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mano", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--sequence", type=Path, required=True)
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
    sequence = json.loads(args.sequence.resolve().read_text(encoding="utf-8"))["sequence"]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for frame in sequence:
        theta = torch.tensor(frame["theta_hand_pose"], dtype=torch.float64)
        with torch.no_grad():
            result = mano(
                betas=beta[None], hand_pose=theta[None],
                global_orient=torch.zeros(1, 3, dtype=torch.float64),
                transl=torch.zeros(1, 3, dtype=torch.float64),
            )
        vertices = scale * (result.vertices[0].numpy() @ rotation.T) + translation
        filename = f"mano_r_{frame['frame_id']:03d}.obj"
        export_obj(out / filename, vertices, mano_data["f"])
        records.append({"frame_id": frame["frame_id"], "name": frame["name"], "obj": filename,
                        "vertex_count": int(vertices.shape[0])})
    (out / "mesh_info.json").write_text(json.dumps({
        "version": "v16_dynamic_R_sequence_meshes",
        "status": "generated_dynamic_MANO_R_sequence_meshes",
        "source_model_manifest": str(manifest_path),
        "source_sequence": str(args.sequence.resolve()),
        "records": records,
        "limitation": "The sequence is synthetic interpolation, not camera data.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "generated_dynamic_MANO_R_sequence_meshes", "frame_count": len(records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
