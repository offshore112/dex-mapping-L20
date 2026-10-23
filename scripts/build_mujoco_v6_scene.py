"""Build and validate a MuJoCo v6 static scene with L20 + fitted MANO-R."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def render(model, data, path, title, qpos=None):
    import mujoco
    from PIL import Image, ImageDraw

    if qpos is not None:
        data.qpos[:] = 0
        for name, value in qpos.items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
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
    image = Image.fromarray(renderer.render())
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 480, 32), fill=(255, 255, 255, 220))
    draw.text((12, 8), title, fill=(25, 45, 65))
    image.save(path)
    renderer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--mano-obj", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import mujoco

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    mesh_dir = out / "meshes"
    shutil.copytree(args.urdf.resolve().parent / "meshes", mesh_dir, dirs_exist_ok=True)
    obj_copy = out / "mano_r_fitted.obj"
    shutil.copy2(args.mano_obj.resolve(), obj_copy)

    spec = mujoco.MjSpec.from_file(str(args.urdf.resolve()))
    xml = spec.to_xml()
    # MjSpec currently drops newly added children when serializing a URDF-derived
    # tree, so append the visual asset/body explicitly to the generated MJCF.
    xml = xml.replace(
        "</asset>",
        '    <mesh name="mano_r_fitted" file="mano_r_fitted.obj"/>\n  </asset>',
        1,
    )
    xml = xml.replace(
        "</worldbody>",
        '    <body name="mano_r_visual" pos="0 0 0">\n'
        '      <geom name="mano_r_surface" type="mesh" mesh="mano_r_fitted" '
        'rgba="0.12 0.48 0.78 0.38" contype="0" conaffinity="0"/>\n'
        '    </body>\n  </worldbody>',
        1,
    )
    scene_path = out / "mujoco_l20_mano_r_v6.xml"
    scene_path.write_text(xml, encoding="utf-8")

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    model_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    geom_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(model.ngeom)]
    mano_geom_id = model.ngeom - 1
    render(model, data, out / "mujoco_v6_zero.png", "v6 MuJoCo | L20 zero pose + MANO-R", {})
    index_joint = "index_pip"
    pulse = {index_joint: 0.5} if index_joint in model_names else {}
    render(model, data, out / "mujoco_v6_index_pip_0p5.png",
           "v6 MuJoCo | index_pip=0.5 rad + MANO-R", pulse)
    check = {
        "version": "v6_mujoco_static_import",
        "status": "passed_static_model_load_and_offscreen_render",
        "mujoco_version": mujoco.__version__,
        "urdf_source": str(args.urdf.resolve()),
        "mano_obj_source": str(args.mano_obj.resolve()),
        "scene": str(scene_path),
        "body_count": int(model.nbody),
        "joint_count": int(model.njnt),
        "geom_count": int(model.ngeom),
        "joint_names": model_names,
        "geom_names": geom_names,
        "mano_visual_geom_present": bool(model.geom_type[mano_geom_id] == mujoco.mjtGeom.mjGEOM_MESH),
        "mano_visual_geom_id": int(mano_geom_id),
        "mano_visual_collision_disabled": bool(model.geom_contype[mano_geom_id] == 0 and model.geom_conaffinity[mano_geom_id] == 0),
        "root_frame": "hand_base_link",
        "unit": "m/rad",
        "dynamic_mano_note": "OBJ is a static fitted visual mesh in v6; framewise MANO vertex updates are a later step.",
    }
    (out / "mujoco_v6_check.json").write_text(json.dumps(check, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": check["status"], "scene": str(scene_path),
                      "bodies": model.nbody, "joints": model.njnt, "geoms": model.ngeom,
                      "mano_geom": check["mano_visual_geom_present"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
