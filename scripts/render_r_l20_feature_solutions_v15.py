"""Render v15 feature-level q solutions in a MuJoCo L20+R scene."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import mujoco
    from PIL import Image, ImageDraw

    scene = args.scene.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(args.metrics.resolve().read_text(encoding="utf-8"))
    model = mujoco.MjModel.from_xml_path(str(scene))
    render_info = []
    for result in metrics["results"]:
        data = mujoco.MjData(model)
        data.qpos[:] = 0
        for name, value in result["q_full_rad"].items():
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
        image = Image.fromarray(renderer.render()).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 480, 34), fill="white")
        pair = result.get("tip_pair")
        task = " + ".join(pair) if pair else "open"
        draw.text((12, 8), f"v15 feature q | {result['name']} | {task}", fill=(25, 45, 65))
        filename = f"{result['name']}.png"
        image.save(out / filename)
        renderer.close()
        render_info.append({
            "action": result["name"],
            "image": filename,
            "tip_pair": pair,
            "note": "R surface is static visual reference; L20 is driven by feature-level q solution.",
        })
    (out / "render_info.json").write_text(json.dumps({
        "version": "v15_R_L20_feature_solution_render",
        "status": "passed_feature_solution_render",
        "renders": render_info,
        "limitations": [
            "Images are a visual q-level check, not contact/collision or actuator validation.",
            "The natural R surface and L20 mechanics are intentionally separate objects.",
        ],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed_feature_solution_render", "renders": render_info}, ensure_ascii=False))


if __name__ == "__main__":
    main()
