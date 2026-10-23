"""Render dynamic MANO-R action surfaces together with v15 L20 q solutions."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def export_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    lines = ["# Dynamic v15 MANO-R surface; metres"]
    lines.extend("v " + " ".join(f"{float(x):.10f}" for x in row) for row in vertices)
    lines.extend("f " + " ".join(str(int(x) + 1) for x in row) for row in faces)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mano", type=Path, help="MANO pickle; required when --obj-dir is absent")
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--scene-template", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--obj-dir", type=Path, help="pre-generated action OBJ directory; allows MuJoCo-only environment")
    args = parser.parse_args()

    import mujoco
    from PIL import Image, ImageDraw
    mano = None
    faces = None
    if args.obj_dir is None:
        if args.mano is None:
            raise ValueError("Provide --mano or --obj-dir")
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
        faces = np.asarray(mano_data["f"], dtype=int)

    template_path = args.scene_template.resolve()
    template = template_path.read_text(encoding="utf-8")
    template_mesh = template_path.parent / "mano_r_fitted.obj"
    if not template_mesh.is_file():
        raise FileNotFoundError(template_mesh)
    actions = json.loads(args.actions.resolve().read_text(encoding="utf-8"))["actions"]
    metrics = {item["name"]: item for item in json.loads(args.metrics.resolve().read_text(encoding="utf-8"))["results"]}
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # MuJoCo on Windows is sensitive to non-ASCII paths.  The caller should
    # use an ASCII output root for scene compilation; rendered PNGs can later
    # be copied into the human-readable experiment directory.
    if any(ord(char) > 127 for char in str(out)):
        raise ValueError("Use an ASCII output path for MuJoCo dynamic scenes")
    mesh_source = template_path.parent / "meshes"
    mesh_target = out / "meshes"
    if mesh_source.is_dir():
        shutil.copytree(mesh_source, mesh_target, dirs_exist_ok=True)

    render_info = []
    for action in actions:
        name = action["name"]
        if args.obj_dir is None:
            theta = torch.tensor(action["theta_hand_pose"], dtype=torch.float64)
            with torch.no_grad():
                result = mano(
                    betas=beta[None], hand_pose=theta[None],
                    global_orient=torch.zeros(1, 3, dtype=torch.float64),
                    transl=torch.zeros(1, 3, dtype=torch.float64),
                )
            vertices = scale * (result.vertices[0].numpy() @ rotation.T) + translation
            obj_path = out / f"mano_r_{name}.obj"
            export_obj(obj_path, vertices, faces)
        else:
            obj_path = args.obj_dir.resolve() / f"mano_r_{name}.obj"
            if not obj_path.is_file():
                raise FileNotFoundError(obj_path)
        scene_path = out / f"scene_{name}.xml"
        xml = template.replace("mano_r_fitted.obj", obj_path.as_posix())
        scene_path.write_text(xml, encoding="utf-8")
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        data = mujoco.MjData(model)
        for joint_name, value in metrics[name]["q_full_rad"].items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if jid >= 0:
                data.qpos[model.jnt_qposadr[jid]] = value
        mujoco.mj_forward(model, data)
        renderer = mujoco.Renderer(model, height=480, width=480)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.azimuth = -72
        camera.elevation = 18
        camera.distance = 0.48
        camera.lookat[:] = [0.0, 0.0, 0.13]
        renderer.update_scene(data, camera=camera)
        image = Image.fromarray(renderer.render()).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 480, 34), fill="white")
        pair = action.get("tip_pair")
        task = " + ".join(pair) if pair else "open"
        draw.text((12, 8), f"v15 dynamic R + q | {name} | {task}", fill=(25, 45, 65))
        filename = f"{name}.png"
        image.save(out / filename)
        renderer.close()
        render_info.append({"action": name, "scene": str(scene_path), "mesh": str(obj_path), "image": filename,
                            "tip_pair": pair, "mj_ncon": int(data.ncon)})
    (out / "render_info.json").write_text(json.dumps({
        "version": "v15_dynamic_R_L20_feature_solution_render",
        "status": "passed_dynamic_R_surface_and_q_solution_render",
        "renders": render_info,
        "limitations": [
            "The image is a visual mapping check; contact/collision is not interpreted as success.",
            "The URDF collision geometry currently has overlapping contacts and needs a separate cleaned collision descriptor.",
        ],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed_dynamic_R_surface_and_q_solution_render", "renders": render_info}, ensure_ascii=False))


if __name__ == "__main__":
    main()
