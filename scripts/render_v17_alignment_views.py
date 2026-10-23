"""Render v17 overlay and separated R/L20 views for key sequence frames."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def render_one(mujoco, Image, ImageDraw, scene_path: Path, q_full: dict, title: str,
               output: Path, lookat: list[float], distance: float) -> None:
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    for name, value in q_full.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = value
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=480, width=480)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = -180
    camera.elevation = 10
    camera.distance = distance
    camera.lookat[:] = lookat
    renderer.update_scene(data, camera=camera)
    image = Image.fromarray(renderer.render()).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 480, 34), fill="white")
    draw.text((12, 9), title, fill=(25, 45, 65))
    image.save(output)
    renderer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-template", type=Path, required=True)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--obj-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", default="v17")
    args = parser.parse_args()

    import mujoco
    from PIL import Image, ImageDraw

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(ord(char) > 127 for char in str(out)):
        raise ValueError("Use an ASCII output path for MuJoCo alignment rendering")
    template_path = args.scene_template.resolve()
    template = template_path.read_text(encoding="utf-8")
    if (template_path.parent / "meshes").is_dir():
        shutil.copytree(template_path.parent / "meshes", out / "meshes", dirs_exist_ok=True)
    sequence = json.loads(args.sequence.resolve().read_text(encoding="utf-8"))["sequence"]
    metrics = json.loads(args.metrics.resolve().read_text(encoding="utf-8"))["results"]
    by_id = {int(item["frame_id"]): item for item in metrics}
    selected = [0, 8, 16]
    overlay_images, separate_images, records = [], [], []
    for frame_id in selected:
        frame = sequence[frame_id]
        metric = by_id[frame_id]
        obj_path = args.obj_dir.resolve() / f"mano_r_{frame_id:03d}.obj"
        if not obj_path.is_file():
            raise FileNotFoundError(obj_path)
        scene_overlay = out / f"scene_{frame_id:03d}_overlay.xml"
        scene_overlay.write_text(template.replace("mano_r_fitted.obj", obj_path.as_posix()), encoding="utf-8")
        overlay_path = out / f"frame_{frame_id:03d}_overlay.png"
        pair = frame.get("tip_pair")
        task = "+".join(pair) if pair else "open"
        render_one(mujoco, Image, ImageDraw, scene_overlay, metric["q_full_rad"],
                   f"{args.version} overlay | frame {frame_id:02d} | {task}", overlay_path,
                   [0.0, 0.0, 0.13], 0.48)
        separate_xml = template.replace("mano_r_fitted.obj", obj_path.as_posix())
        separate_xml = separate_xml.replace('<body name="mano_r_visual" pos="0 0 0">',
                                              '<body name="mano_r_visual" pos="0 -0.30 0">')
        scene_separate = out / f"scene_{frame_id:03d}_separate.xml"
        scene_separate.write_text(separate_xml, encoding="utf-8")
        separate_path = out / f"frame_{frame_id:03d}_separate.png"
        render_one(mujoco, Image, ImageDraw, scene_separate, metric["q_full_rad"],
                   f"{args.version} separate | blue R / gray L20 | frame {frame_id:02d}", separate_path,
                   [0.0, -0.15, 0.13], 0.95)
        overlay_images.append(Image.open(overlay_path).convert("RGB"))
        separate_images.append(Image.open(separate_path).convert("RGB"))
        records.append({"frame_id": frame_id, "overlay": overlay_path.name, "separate": separate_path.name,
                        "tip_pair": pair})
    overlay_sheet = Image.new("RGB", (480 * 3, 480), "white")
    separate_sheet = Image.new("RGB", (480 * 3, 480), "white")
    for index, image in enumerate(overlay_images):
        overlay_sheet.paste(image, (index * 480, 0))
    for index, image in enumerate(separate_images):
        separate_sheet.paste(image, (index * 480, 0))
    overlay_sheet.save(out / "v17_overlay_contact_sheet.png")
    separate_sheet.save(out / "v17_separate_contact_sheet.png")
    (out / "render_info.json").write_text(json.dumps({
        "version": f"{args.version}_R_L20_alignment_views",
        "status": "passed_overlay_and_separated_view_render",
        "selected_frames": selected,
        "records": records,
        "visual_convention": {
            "overlay": "natural blue MANO-R surface and gray L20 geometry share the root for alignment inspection",
            "separate": "blue R is translated -0.30 m along the camera-horizontal palm axis for side-by-side inspection; L20 remains at the root",
        },
        "limitations": [
            "Images show synthetic sequence endpoints, not camera playback.",
            "The separated view is a display translation only and does not change q or feature optimization.",
        ],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed_overlay_and_separated_view_render", "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
