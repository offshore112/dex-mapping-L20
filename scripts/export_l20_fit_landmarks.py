"""Export the 21-point L20 target used by the initial MANO-R fit.

The target is the L20 URDF zero pose. Joint points are joint-frame origins;
tip and wrist points come from the v4 mesh reference-point file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from l20_kinematics import HandURDF


FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")
FINGER_JOINTS = {
    "thumb": ("thumb_cmc_pitch", "thumb_mcp", "thumb_ip"),
    "index": ("index_mcp_pitch", "index_pip", "index_dip"),
    "middle": ("middle_mcp_pitch", "middle_pip", "middle_dip"),
    "ring": ("ring_mcp_pitch", "ring_pip", "ring_dip"),
    "pinky": ("pinky_mcp_pitch", "pinky_pip", "pinky_dip"),
}


def add_point(points, name, xyz, source, link=None, joint=None):
    points.append({
        "name": name,
        "xyz_m": [float(v) for v in np.asarray(xyz, dtype=float)],
        "source": source,
        **({"link": link} if link else {}),
        **({"joint": joint} if joint else {}),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--reference-points", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    robot = HandURDF(args.urdf.resolve())
    transforms, frames, full = robot.fk({})
    refs = json.loads(args.reference_points.read_text(encoding="utf-8"))
    ref_points = refs["points"]

    points = []
    add_point(points, "wrist", ref_points["wrist_ref"]["zero_pose_root_xyz_m"],
              "v4_mesh_reference_point", link="hand_base_link")
    for finger in FINGER_ORDER:
        suffixes = ("cmc", "mcp", "ip") if finger == "thumb" else ("mcp", "pip", "dip")
        for suffix, joint_name in zip(suffixes, FINGER_JOINTS[finger]):
            if joint_name not in frames:
                raise KeyError(f"Joint frame missing from FK: {joint_name}")
            add_point(points, f"{finger}_{suffix}", frames[joint_name][:3, 3],
                      "urdf_joint_frame_origin", joint=joint_name)
        tip_name = f"{finger}_tip"
        add_point(points, tip_name, ref_points[tip_name]["zero_pose_root_xyz_m"],
                  "v4_mesh_reference_point", link=f"{finger}_distal")

    if len(points) != 21 or [p["name"] for p in points] != [
        "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
        "index_mcp", "index_pip", "index_dip", "index_tip",
        "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
        "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
        "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
    ]:
        raise AssertionError("Unexpected L20 landmark order")

    record = {
        "version": "v5_l20_fit_target",
        "status": "generated_for_initial_simulation_fit",
        "coordinate_frame": "hand_base_link",
        "unit": "m",
        "reference_pose": "all independent URDF coordinates zero; mimic coordinates expanded by URDF",
        "point_order": [p["name"] for p in points],
        "points": points,
        "full_joint_values_rad": {k: float(v) for k, v in full.items()},
        "source": {
            "urdf": str(args.urdf.resolve()),
            "reference_points": str(args.reference_points.resolve()),
            "mcp_proxy_rule": "mcp_pitch joint frame for the four fingers",
            "thumb_base_proxy_rule": "thumb_cmc_pitch joint frame",
        },
        "limitations": [
            "The target is CAD/URDF geometry in the zero pose, not a hardware measurement.",
            "Joint-frame origins are mechanical proxies for MANO joints.",
            "Tip points are distal-cap centers, not measured contact centers.",
        ],
    }
    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} with {len(points)} points")
    for p in points:
        print(f"{p['name']:12s} {np.round(np.asarray(p['xyz_m']) * 1000, 3)} mm")


if __name__ == "__main__":
    main()
