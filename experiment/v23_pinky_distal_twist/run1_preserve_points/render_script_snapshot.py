"""Render old/new fixed-R surfaces with one camera and validate v23 geometry."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from view_fixed_r_mujoco_v22 import create_scene, load_geometry, scene_check, update_frame


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def summary(values):
    return {"mean": float(np.mean(values)), "rmse": float(np.sqrt(np.mean(values ** 2))),
            "max": float(np.max(values)), "p95": float(np.percentile(values, 95))}


def angle(a, b):
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=-1), -1, 1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("experiment/v23_pinky_distal_twist/run1_preserve_points"))
    parser.add_argument("--old", type=Path, default=Path("experiment/v22_personal_fixed_r/motion_geometry.npz"))
    parser.add_argument("--fps", type=float, default=8)
    parser.add_argument("--neutral-obj", type=Path, default=Path("experiment/v12_R_L20_canonical_human/r_l20_surface.obj"))
    args = parser.parse_args()
    run = args.run.resolve()
    out = run / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    if not 0 < args.fps <= 30:
        raise ValueError("fps must be in (0,30]")
    old = load_geometry(args.old)
    new = load_geometry(run / "fixed_r/motion_geometry.npz")
    if not np.array_equal(old["frame_ids"], new["frame_ids"]) or not np.array_equal(old["timestamps_s"], new["timestamps_s"]):
        raise ValueError("Old/new frame IDs or timestamps differ")
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    if digest(Path(config["input"])) != config["input_sha256"] or digest(args.old) != config["baseline_geometry_sha256"]:
        raise ValueError("Baseline changed after the audit")
    source = json.loads(Path(config["input"]).read_text(encoding="utf-8"))
    corrected = json.loads((run / "mano_sequence_v23.json").read_text(encoding="utf-8"))
    changed_fields = {"theta_hand_pose", "fitted_landmarks_m", "point_error_mm", "phase"}
    for before, after in zip(source["frames"], corrected["frames"]):
        for key in before:
            if key not in changed_fields and before[key] != after[key]:
                raise ValueError(f"Unintended source field change: {key}")
    # Scalar rotation convention and semantic ordering have already been
    # checked in the fitting audit. Here verify both complete display skins.
    comparison = dict(new)
    comparison["source_vertices_m"] = old["r_vertices_m"]
    comparison["source_points_m"] = old["r_points_m"]
    comparison["camera_normal"] = old["camera_normal"]
    np.savez_compressed(out / "old_new_geometry.npz", **comparison)
    model, data, records, camera = create_scene(comparison, "separate", out / "scene.xml")
    skin_check = scene_check(model, data, records, comparison)
    from PIL import Image, ImageDraw, ImageFont
    import mujoco

    width, height = 1200, 700
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 21)
    small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
    times = new["timestamps_s"] - new["timestamps_s"][0]
    old_points, new_points = old["r_points_m"], new["r_points_m"]
    tip_change = np.linalg.norm(new_points[:, 20] - old_points[:, 20], axis=1) * 1000
    direction_change = angle(new_points[:, 20] - new_points[:, 19], old_points[:, 20] - old_points[:, 19])
    metric = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    before_twist = np.load(run / "audit_geometry.npz")["before_twist_deg"]
    after_twist = np.load(run / "audit_geometry.npz")["after_twist_deg"]
    renderer = mujoco.Renderer(model, height=height, width=width)
    png = out / "frames"
    png.mkdir(exist_ok=True)
    initial_azimuth, initial_elevation = float(camera.azimuth), float(camera.elevation)
    half_separation = float(np.linalg.norm(records[0]["offset"] - records[1]["offset"]) / 2)

    def render(index, suffix="palm", azimuth_offset=0, elevation_offset=0):
        camera.azimuth = initial_azimuth + azimuth_offset
        camera.elevation = initial_elevation + elevation_offset
        # Keep the display-only separation horizontal for every view. A
        # fixed world offset would overlap the two surfaces in the side view.
        azimuth = np.radians(camera.azimuth)
        right = np.array([-np.sin(azimuth), np.cos(azimuth), 0.])
        records[0]["offset"] = right * half_separation
        records[1]["offset"] = -right * half_separation
        update_frame(model, data, records, index)
        renderer.update_scene(data, camera=camera)
        im = Image.fromarray(renderer.render()).convert("RGB")
        draw = ImageDraw.Draw(im)
        draw.rectangle((0, 0, width, 70), fill=(246, 248, 251))
        draw.text((16, 6), f"Pinky DIP rotation audit | frame {new['frame_ids'][index]} | t={times[index]:.2f}s | {suffix}", font=font, fill=(20, 35, 50))
        draw.text((16, 38), f"GOLD: v22 original  |  source-pose twist {before_twist[index]:+.1f} deg", font=small, fill=(145, 83, 12))
        draw.text((625, 38), f"BLUE: v23 candidate  |  source-pose twist {after_twist[index]:+.1f} deg", font=small, fill=(20, 92, 160))
        draw.rectangle((0, height - 32, width, height), fill=(246, 248, 251))
        draw.text((16, height - 27), f"Same R shape, scale and camera. Only pinky DIP pose changes. R tip displacement: {tip_change[index]:.3f} mm. Neutral-twist prior; no nail truth.", font=small, fill=(45, 63, 82))
        return im

    try:
        selected_ids = [0, 5, 30, 60, 120, 190, 250, 383, 773]
        by_id = {int(fid): i for i, fid in enumerate(new["frame_ids"])}
        for frame_id in selected_ids:
            if frame_id not in by_id:
                continue
            i = by_id[frame_id]
            panels = [render(i), render(i, "oblique", 35, -20), render(i, "side", 75, 0)]
            sheet = Image.new("RGB", (width, height * 3))
            for row, panel in enumerate(panels):
                sheet.paste(panel, (0, row * height))
            sheet.save(out / f"frame_{frame_id:04d}_multiview.png")
        grid = np.arange(0, times[-1], 1 / args.fps)
        indices = np.unique(np.clip(np.searchsorted(times, grid, side="right") - 1, 0, len(times) - 1))
        indices = np.unique(np.r_[indices, len(times) - 1])
        images = []
        for index in indices:
            im = render(int(index))
            if int(new["frame_ids"][index]) in selected_ids:
                im.save(png / f"frame_{new['frame_ids'][index]:04d}.png")
            images.append(im.quantize(colors=128))
        boundaries = np.r_[times[indices], times[-1] + np.median(np.diff(times))]
        durations = np.maximum(1, np.diff(np.rint(boundaries * 100))) * 10
        images[0].save(out / "old_new_fixed_r.gif", save_all=True, append_images=images[1:],
                       duration=durations.astype(int).tolist(), loop=0, optimize=False)
    finally:
        renderer.close()

    # Neutral surface from the unchanged v12 OBJ, in the same target frame.
    obj = args.neutral_obj
    neutral_v = np.array([[float(v) for v in line.split()[1:]] for line in obj.read_text().splitlines() if line.startswith("v ")])
    neutral = dict(new)
    neutral["source_vertices_m"] = neutral_v[None]
    neutral["r_vertices_m"] = neutral_v[None]
    neutral["source_points_m"] = new["neutral_r_points_m"][None]
    neutral["r_points_m"] = new["neutral_r_points_m"][None]
    neutral["timestamps_s"] = np.array([0.])
    neutral["frame_ids"] = np.array([-1])
    nm, nd, nr, nc = create_scene(neutral, "r-only", out / "neutral_scene.xml")
    neutral_check = scene_check(nm, nd, nr, neutral)
    renderer = mujoco.Renderer(nm, height=height, width=width)
    try:
        update_frame(nm, nd, nr, 0)
        renderer.update_scene(nd, camera=nc)
        im = Image.fromarray(renderer.render())
        draw = ImageDraw.Draw(im)
        draw.rectangle((0, 0, width, 65), fill=(246, 248, 251))
        draw.text((16, 8), "Unchanged v12 fixed R | beta_R = 0 | hand_pose = 0", font=font, fill=(20, 35, 50))
        draw.text((16, 37), "Neutral mesh has no added pinky DIP twist. Dynamic surface follows fitted rotations.", font=small, fill=(45, 63, 82))
        im.save(out / "neutral_fixed_r.png")
    finally:
        renderer.close()

    validation = {
        "version": "v23_pinky_distal_twist", "status": "passed_offline_geometry_and_display_checks",
        "all774_fields_preserved_except": sorted(changed_fields),
        "original_v21_and_v22_hashes_unchanged": True,
        "frame_count": len(times), "r_all21_point_change_mm": summary(np.linalg.norm(new_points - old_points, axis=2) * 1000),
        "r_pinky_tip_change_mm": summary(tip_change), "r_pinky_tip_direction_change_deg": summary(direction_change),
        "r_pinky_tip_change_first60_mm": summary(tip_change[:60]),
        "r_other16_points_max_change_mm": float(np.max(np.linalg.norm(new_points[:, :17] - old_points[:, :17], axis=2)) * 1000),
        "pinky_mcp_pip_dip_centers_max_change_mm": float(np.max(np.linalg.norm(new_points[:, 17:20] - old_points[:, 17:20], axis=2)) * 1000),
        "mujoco_old_and_new": skin_check, "neutral_skin": neutral_check,
        "rendered_preview_frames": len(indices), "preview_duration_s": float(durations.sum() / 1000),
        "render_script_sha256": digest(Path(__file__)),
        "baseline_point_rmse_mm": metric["all21_observation_point_error_mm_before"]["rmse"],
        "candidate_point_rmse_mm": metric["all21_observation_point_error_mm_after"]["rmse"],
        "scope": "single recorded sequence; visual kinematics, no physics or nail-orientation ground truth",
    }
    write_json(out / "validation.json", validation)
    print(json.dumps(validation), flush=True)


if __name__ == "__main__":
    main()
