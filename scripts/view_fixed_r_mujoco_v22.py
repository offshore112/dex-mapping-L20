"""Play/render v22's exact MANO surface in MuJoCo using visual skins.

One mocap body per surface vertex provides an exact visual deformation.
There are no simulated joints, actuators or collision geoms. The compiled
model is never mutated: playback updates mjData.mocap_pos and calls forward.
Run with the mano-l20-sim environment, not the torch fitting environment.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def load_geometry(path):
    with np.load(path, allow_pickle=False) as a:
        values = {key: a[key].copy() for key in a.files}
    for key in ("source_vertices_m", "r_vertices_m", "r_points_m", "source_points_m", "timestamps_s"):
        if not np.isfinite(values[key]).all():
            raise ValueError(f"Nonfinite {key}")
    return values


def number_list(array):
    return " ".join(f"{float(value):.9g}" for value in np.asarray(array).ravel())


def create_scene(geometry, view, scene_path):
    import mujoco

    normal = geometry["camera_normal"].astype(float)
    normal /= np.linalg.norm(normal)
    right = np.cross([0.0, 0.0, 1.0], normal)
    right /= np.linalg.norm(right)
    source, target = geometry["source_vertices_m"], geometry["r_vertices_m"]
    spans = np.ptp(np.concatenate([source.reshape(-1, 3), target.reshape(-1, 3)]), axis=0)
    separation = max(float(np.max(spans)) * 0.85, 0.22)
    shift_source = right * separation / 2 if view == "separate" else np.zeros(3)
    shift_target = -right * separation / 2 if view == "separate" else np.zeros(3)
    bodies, skins, records = [], [], []
    for name, vertices, offset, rgba in (
            ("source", source, shift_source, "0.90 0.58 0.23 1"),
            ("R", target, shift_target, "0.12 0.48 0.78 1")):
        if view == "r-only" and name == "source":
            continue
        initial = vertices[0].astype(float) + offset
        bones = []
        for i, position in enumerate(initial):
            body_name = f"{name}_v{i}"
            pos = number_list(position)
            bodies.append(f'<body name="{body_name}" mocap="true" pos="{pos}"/>')
            bones.append(f'<bone body="{body_name}" bindpos="{pos}" bindquat="1 0 0 0" vertid="{i}" vertweight="1"/>')
        faces = " ".join(str(int(v)) for v in geometry["faces"].ravel())
        skins.append(f'<skin name="{name}_surface" group="0" rgba="{rgba}" inflate="0" vertex="{number_list(initial)}" face="{faces}">' + "".join(bones) + '</skin>')
        records.append({"name": name, "vertices": vertices, "offset": offset, "count": len(initial)})
    xml = ('<mujoco model="v22_personal_fixed_R"><compiler angle="radian"/>'
           '<option gravity="0 0 0"/>'
           '<visual><global offwidth="1200" offheight="700"/><headlight ambient="0.5 0.5 0.5" diffuse="0.7 0.7 0.7" specular="0.2 0.2 0.2"/></visual>'
           '<worldbody>' + "".join(bodies) + '</worldbody><deformable>' + "".join(skins) + '</deformable></mujoco>')
    scene_path.write_text(xml, encoding="utf-8")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    for record in records:
        record["mocap_ids"] = np.asarray([
            model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{record['name']}_v{i}")]
            for i in range(record["count"])
        ])
    all_positions = np.concatenate([record["vertices"].reshape(-1, 3) + record["offset"] for record in records])
    center = (all_positions.min(axis=0) + all_positions.max(axis=0)) / 2
    radius = float(np.max(np.linalg.norm(all_positions - center, axis=1)))
    camera = mujoco.MjvCamera()
    camera.azimuth = float(np.degrees(np.arctan2(normal[1], normal[0])))
    camera.elevation = float(np.degrees(np.arcsin(np.clip(normal[2], -1, 1))))
    camera.distance = radius * 2.6
    camera.lookat[:] = center
    return model, data, records, camera


def update_frame(model, data, records, index):
    import mujoco
    for record in records:
        data.mocap_pos[record["mocap_ids"]] = record["vertices"][index] + record["offset"]
    mujoco.mj_forward(model, data)


def scene_check(model, data, records, geometry):
    import mujoco
    scene = mujoco.MjvScene(model, maxgeom=100)
    option = mujoco.MjvOption()
    camera = mujoco.MjvCamera()
    max_error = 0.0
    for index in range(len(geometry["timestamps_s"])):
        update_frame(model, data, records, index)
        mujoco.mjv_updateScene(model, data, option, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
        # skinvert follows skin order; compare the displayed vertices, not
        # just the mocap bodies, against the generated MANO geometry.
        rendered = np.asarray(scene.skinvert).reshape(-1, 3)
        expected = np.concatenate([record["vertices"][index] + record["offset"] for record in records])
        max_error = max(max_error, float(np.max(np.abs(rendered - expected))))
    if max_error > 1e-6:
        raise RuntimeError(f"MuJoCo skin does not match MANO vertices: {max_error} m")
    return {"status": "passed_mujoco_visual_skin_check", "body_count": int(model.nbody),
            "checked_frame_count": len(geometry["timestamps_s"]),
            "joint_count": int(model.njnt), "mocap_count": int(model.nmocap),
            "skin_count": int(model.nskin), "geom_count": int(model.ngeom),
            "skin_vertex_max_abs_error_m": max_error,
            "physics": "not stepped; visual kinematic playback only"}


def render_preview(model, data, records, geometry, camera, output, fps, seconds, label="v22"):
    import mujoco
    from PIL import Image, ImageDraw, ImageFont

    times = geometry["timestamps_s"]
    relative = times - times[0]
    duration = min(float(relative[-1]), seconds) if seconds > 0 else float(relative[-1])
    grid = np.arange(0.0, duration, 1.0 / fps)
    indices = np.searchsorted(relative, grid, side="right") - 1
    indices = np.clip(indices, 0, len(times) - 1)
    png_dir = output / "frames"
    png_dir.mkdir(exist_ok=True)
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    font = ImageFont.truetype(str(font_path), 19) if font_path.is_file() else ImageFont.load_default()
    renderer = mujoco.Renderer(model, height=600, width=1000)
    images, rendered_indices = [], []
    try:
        for index in dict.fromkeys(indices.tolist()):
            update_frame(model, data, records, int(index))
            renderer.update_scene(data, camera=camera)
            image = Image.fromarray(renderer.render()).convert("RGB")
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, 1000, 58), fill=(246, 248, 251))
            draw.text((15, 5), f"{label} | personal motion -> fixed-shape R | t={relative[index]:.2f}s | frame={int(geometry['frame_ids'][index])}", font=font, fill=(20, 35, 50))
            names = [record["name"] for record in records]
            if names == ["source", "R"]:
                draw.text((15, 31), "GOLD: your fitted hand", font=font, fill=(170, 100, 20))
                draw.text((520, 31), "BLUE: fixed R (beta_R = 0)", font=font, fill=(20, 100, 180))
            else:
                draw.text((15, 31), "BLUE: fixed R | recorded sequence playback", font=font, fill=(20, 100, 180))
            image.save(png_dir / f"frame_{int(geometry['frame_ids'][index]):04d}.png")
            images.append(image.quantize(colors=128))
            rendered_indices.append(int(index))
            if len(images) % 50 == 0:
                print(json.dumps({"status": "rendering", "frames": len(images)}), flush=True)
    finally:
        renderer.close()
    # GIF stores centiseconds. Quantize cumulative boundaries, rather than
    # flooring each duration independently (which speeds up long previews).
    boundaries = np.asarray([relative[i] for i in rendered_indices] + [duration])
    ticks = np.rint(boundaries * 100.0).astype(int)
    durations = np.maximum(1, np.diff(ticks)) * 10
    gif_path = output / "personal_to_fixed_r.gif"
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=durations.tolist(), loop=0, optimize=False)
    requested = [int(geometry["frame_ids"][0]), 190, 383, int(geometry["frame_ids"][-1])]
    sheet = Image.new("RGB", (1000, 600 * len(requested)), "white")
    for row, frame_id in enumerate(requested):
        closest = min(rendered_indices, key=lambda i: abs(int(geometry["frame_ids"][i]) - frame_id))
        sheet.paste(images[rendered_indices.index(closest)].convert("RGB"), (0, row * 600))
    sheet.save(output / "comparison_keyframes.png")
    info = {"status": "completed_mujoco_render", "label": label, "frame_count": len(images),
            "source_geometry_frame_count": len(times), "preview_fps": fps,
            "duration_s": duration, "gif_duration_s": float(durations.sum() / 1000.0), "gif": str(gif_path),
            "records": [{"frame_id": int(geometry["frame_ids"][index]),
                         "timestamp_relative_s": float(relative[index])} for index in rendered_indices]}
    (output / "render_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return info


def play(model, data, records, geometry, camera, speed, seconds):
    import mujoco.viewer
    relative = geometry["timestamps_s"] - geometry["timestamps_s"][0]
    duration = float(relative[-1] + np.median(np.diff(relative)))
    print("MuJoCo: recorded personal motion drives fixed R; close the window to exit.", flush=True)
    print("Space: pause/resume; playback repeats. Mouse: rotate/zoom/pan.", flush=True)
    state = {"paused": False, "elapsed": 0.0}
    def key_callback(key):
        if key == 32:
            state["paused"] = not state["paused"]
    started, previous = time.perf_counter(), time.perf_counter()
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.azimuth, viewer.cam.elevation = camera.azimuth, camera.elevation
        viewer.cam.distance = camera.distance
        viewer.cam.lookat[:] = camera.lookat
        while viewer.is_running():
            now = time.perf_counter()
            if seconds > 0 and now - started >= seconds:
                break
            if not state["paused"]:
                state["elapsed"] += (now - previous) * speed
            previous = now
            index = int(np.searchsorted(relative, state["elapsed"] % duration, side="right") - 1)
            with viewer.lock():
                update_frame(model, data, records, max(0, index))
            viewer.sync()
            time.sleep(0.005)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, default=Path("experiment/v22_personal_fixed_r/motion_geometry.npz"))
    parser.add_argument("--view", choices=["r-only", "separate", "overlay"], default="r-only")
    parser.add_argument("--output", type=Path, default=Path("experiment/v22_personal_fixed_r/mujoco"))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--label", default="v22", help="display label in rendered preview; geometry is unchanged")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--seconds", type=float, default=0.0, help="0: entire render or unlimited GUI playback")
    args = parser.parse_args()
    if not 0 < args.fps <= 100 or args.speed <= 0 or args.seconds < 0:
        raise ValueError("fps must be in (0,100], speed positive, seconds nonnegative")
    output = args.output.resolve() / args.view
    if any(ord(c) > 127 for c in str(output)):
        raise ValueError("Use an ASCII MuJoCo output path")
    output.mkdir(parents=True, exist_ok=True)
    geometry = load_geometry(args.motion.resolve())
    model, data, records, camera = create_scene(geometry, args.view, output / "scene.xml")
    check = scene_check(model, data, records, geometry)
    (output / "scene_check.json").write_text(json.dumps(check, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(check), flush=True)
    if args.check_only:
        return
    if args.render:
        result = render_preview(model, data, records, geometry, camera, output, args.fps, args.seconds, args.label)
        print(json.dumps({key: result[key] for key in ("status", "frame_count", "source_geometry_frame_count", "duration_s", "gif")}, ensure_ascii=False), flush=True)
    else:
        play(model, data, records, geometry, camera, args.speed, args.seconds)


if __name__ == "__main__":
    main()
