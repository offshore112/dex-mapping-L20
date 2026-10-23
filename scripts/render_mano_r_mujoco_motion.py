"""Render v7 MANO-R vertex updates inside the v6 MuJoCo scene."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np


def read_obj_vertices(path: Path):
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            values.append([float(v) for v in line.split()[1:4]])
    return np.asarray(values, dtype=float)


def read_obj(path: Path):
    vertices, faces = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            vertices.append([float(v) for v in line.split()[1:4]])
        elif line.startswith("f "):
            faces.append([int(v.split("/")[0]) for v in line.split()[1:4]])
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int)


def write_obj(path: Path, vertices, faces):
    lines = ["# MANO-R frame; metres"]
    lines.extend("v " + " ".join(f"{x:.10f}" for x in v) for v in vertices)
    lines.extend("f " + " ".join(str(int(x)) for x in f) for f in faces)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--baseline-obj", type=Path, required=True)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import mujoco
    from PIL import Image, ImageDraw

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sequence = np.load(args.sequence.resolve())
    _, faces = read_obj(args.baseline_obj.resolve())
    scene_path = args.scene.resolve()
    scene_xml = scene_path.read_text(encoding="utf-8")
    # The frame XML files live under ``out``.  MuJoCo resolves relative mesh
    # paths from the XML file's directory, so carry over the v6 mesh folder.
    source_meshes = scene_path.parent / "meshes"
    if source_meshes.is_dir():
        shutil.copytree(source_meshes, out / "meshes", dirs_exist_ok=True)
    frames = []
    for i, vertices in enumerate(sequence["vertices_m"]):
        frame_name = f"mano_r_frame_{i:03d}.obj"
        write_obj(out / frame_name, vertices, faces)
        frame_xml_path = out / f"scene_frame_{i:03d}.xml"
        frame_xml_path.write_text(scene_xml.replace("mano_r_fitted.obj", frame_name), encoding="utf-8")
        model = mujoco.MjModel.from_xml_path(str(frame_xml_path))
        data = mujoco.MjData(model)
        data.qpos[:] = 0
        mujoco.mj_forward(model, data)
        renderer = mujoco.Renderer(model, height=480, width=480)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.azimuth = -72; camera.elevation = 18; camera.distance = 0.48; camera.lookat[:] = [0, 0, 0.13]
        renderer.update_scene(data, camera=camera)
        image = Image.fromarray(renderer.render()).convert("RGB")
        draw = ImageDraw.Draw(image)
        value = float(sequence["channel_values"][i])
        draw.rectangle((0, 0, 480, 32), fill=(255, 255, 255))
        draw.text((12, 8), f"v7 MuJoCo MANO-R motion | channel increment={value:.3f} rad", fill=(25, 45, 65))
        path = out / f"frame_{i:03d}.png"
        image.save(path)
        frames.append(image.copy())
        renderer.close()
    if frames:
        frames[0].save(out / "mano_r_motion.gif", save_all=True, append_images=frames[1:], duration=160, loop=0)
    (out / "render_info.json").write_text(__import__("json").dumps({
        "version": "v7_mujoco_mano_r_motion",
        "status": "passed_framewise_mesh_update_and_render",
        "frame_count": len(frames),
        "vertex_count": int(sequence["vertices_m"].shape[1]),
        "update_method": "one OBJ and one MuJoCo compile per preview frame",
        "l20_joint_state": "zero pose; L20 was not retargeted in this preview",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"rendered {len(frames)} frames to {out}")


if __name__ == "__main__":
    main()
