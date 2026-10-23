"""Render the v16 MANO-R and L20 q sequence, then make a GIF/contact sheet."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-template", type=Path, required=True)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--obj-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import mujoco
    from PIL import Image, ImageDraw

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(ord(char) > 127 for char in str(out)):
        raise ValueError("Use an ASCII output path for MuJoCo sequence rendering")
    template_path = args.scene_template.resolve()
    template = template_path.read_text(encoding="utf-8")
    mesh_source = template_path.parent / "meshes"
    if mesh_source.is_dir():
        shutil.copytree(mesh_source, out / "meshes", dirs_exist_ok=True)
    sequence = json.loads(args.sequence.resolve().read_text(encoding="utf-8"))["sequence"]
    metric_frames = json.loads(args.metrics.resolve().read_text(encoding="utf-8"))["results"]
    if len(sequence) != len(metric_frames):
        raise ValueError("Sequence target and metric frame counts differ")
    images = []
    records = []
    for frame, metric in zip(sequence, metric_frames):
        frame_id = int(frame["frame_id"])
        obj_path = args.obj_dir.resolve() / f"mano_r_{frame_id:03d}.obj"
        if not obj_path.is_file():
            raise FileNotFoundError(obj_path)
        scene_path = out / f"scene_{frame_id:03d}.xml"
        scene_path.write_text(template.replace("mano_r_fitted.obj", obj_path.as_posix()), encoding="utf-8")
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        data = mujoco.MjData(model)
        for joint_name, value in metric["q_full_rad"].items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if jid >= 0:
                data.qpos[model.jnt_qposadr[jid]] = value
        mujoco.mj_forward(model, data)
        renderer = mujoco.Renderer(model, height=480, width=480)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.azimuth = -180
        camera.elevation = 10
        camera.distance = 0.48
        camera.lookat[:] = [0.0, 0.0, 0.13]
        renderer.update_scene(data, camera=camera)
        image = Image.fromarray(renderer.render()).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 480, 34), fill="white")
        pair = frame.get("tip_pair")
        task = " + ".join(pair) if pair else "open"
        draw.text((12, 8), f"v16 frame {frame_id:02d} | {frame['phase']} | {task}", fill=(25, 45, 65))
        filename = f"frame_{frame_id:03d}.png"
        image.save(out / filename)
        images.append(image.copy())
        renderer.close()
        records.append({"frame_id": frame_id, "image": filename, "phase": frame["phase"],
                        "tip_pair": pair, "raw_contact_count": int(data.ncon)})
    images[0].save(out / "sequence.gif", save_all=True, append_images=images[1:], duration=140, loop=0)
    sheet = Image.new("RGB", (480 * 4, 480 * 5), "white")
    for index, image in enumerate(images):
        sheet.paste(image, ((index % 4) * 480, (index // 4) * 480))
    sheet.save(out / "sequence_contact_sheet.png")
    (out / "render_info.json").write_text(json.dumps({
        "version": "v16_R_L20_sequence_render",
        "status": "passed_sequence_render_and_gif",
        "frame_count": len(records),
        "records": records,
        "limitations": [
            "The GIF visualizes synthetic MANO-R and q outputs; it is not camera playback.",
            "Contact counts are diagnostic only because the current URDF collision meshes overlap.",
        ],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed_sequence_render_and_gif", "frame_count": len(records),
                      "gif": str(out / 'sequence.gif'), "contact_sheet": str(out / 'sequence_contact_sheet.png')}, ensure_ascii=False))


if __name__ == "__main__":
    main()
