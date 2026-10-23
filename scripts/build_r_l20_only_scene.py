"""Build a MuJoCo scene containing only the shared R_L20 MANO surface."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--azimuth", type=float, default=-180.0)
    parser.add_argument("--elevation", type=float, default=10.0)
    parser.add_argument("--version", default="v10", help="artifact version label")
    args = parser.parse_args()

    import mujoco
    from PIL import Image, ImageDraw

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    obj = out / "r_l20_surface.obj"
    shutil.copy2(args.obj.resolve(), obj)
    xml = """<mujoco model="r_l20_only">
  <compiler angle="radian"/>
  <asset>
    <mesh name="r_l20_surface" file="r_l20_surface.obj"/>
  </asset>
  <worldbody>
    <geom name="r_l20_surface" type="mesh" mesh="r_l20_surface"
          rgba="0.12 0.48 0.78 1" contype="0" conaffinity="0"/>
  </worldbody>
</mujoco>
"""
    scene = out / "r_l20_only.xml"
    scene.write_text(xml, encoding="utf-8")
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=480, width=480)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = args.azimuth
    camera.elevation = args.elevation
    camera.distance = 0.48
    camera.lookat[:] = [0.0, -0.04, 0.13]
    renderer.update_scene(data, camera=camera)
    image = Image.fromarray(renderer.render()).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 480, 32), fill=(255, 255, 255))
    draw.text((12, 8), "R_L20 only | independent virtual hand", fill=(25, 45, 65))
    image.save(out / "r_l20_only.png")
    renderer.close()
    check = {
        "version": f"{args.version}_shared_R_L20_only_scene",
        "status": "passed_r_only_mujoco_load_and_render",
        "scene": str(scene),
        "body_count": int(model.nbody),
        "joint_count": int(model.njnt),
        "geom_count": int(model.ngeom),
        "surface_collision_disabled": bool(model.geom_contype[0] == 0 and model.geom_conaffinity[0] == 0),
        "note": "R is rendered independently; L20 is not embedded in this scene.",
    }
    (out / "r_l20_only_check.json").write_text(json.dumps(check, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(check, ensure_ascii=False))


if __name__ == "__main__":
    main()
