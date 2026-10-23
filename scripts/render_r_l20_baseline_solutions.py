"""Render q solutions from the v14 R->L20 baseline in MuJoCo."""
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
    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    render_info = []
    for result in metrics["task_only_pair_diagnostics"]:
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
        draw.rectangle((0, 0, 480, 32), fill="white")
        label = "+".join(result["pair"])
        draw.text((12, 8), f"v14 task-only | {label}", fill=(25, 45, 65))
        filename = result["pair"][0] + "_" + result["pair"][1] + ".png"
        image.save(out / filename)
        renderer.close()
        render_info.append({"pair": result["pair"], "image": filename, "joint_count": len(joint_names)})
    (out / "render_info.json").write_text(json.dumps({
        "version": "v14_R_L20_task_only_render",
        "status": "passed_task_only_solution_render",
        "renders": render_info,
        "note": "Images show q-level point solutions; no contact/collision or actuator validation.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed_task_only_solution_render", "renders": render_info}, ensure_ascii=False))


if __name__ == "__main__":
    main()
