"""Create initial L20 mesh reference points for simulation-only MANO-R fitting.

The output is deliberately a geometric convention, not a hardware calibration.
It uses the local link coordinates from the URDF and area-weighted centroids of
surface triangles in thin end-cap bands.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

from l20_kinematics import HandURDF


TIP_LINKS = {
    "thumb_tip": "thumb_distal",
    "index_tip": "index_distal",
    "middle_tip": "middle_distal",
    "ring_tip": "ring_distal",
    "pinky_tip": "pinky_distal",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def area_weighted_cap_centroid(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    axis: int,
    side: str,
    band_m: float,
) -> tuple[np.ndarray, int, float, float]:
    """Return a surface centroid for a thin min/max coordinate cap."""
    values = vertices[:, axis]
    extreme = float(values.min() if side == "min" else values.max())
    if side == "min":
        selected = values <= extreme + band_m
    elif side == "max":
        selected = values >= extreme - band_m
    else:
        raise ValueError(f"Unsupported cap side: {side}")

    face_mask = selected[faces].all(axis=1)
    triangles = vertices[faces[face_mask]]
    if len(triangles) == 0:
        raise ValueError("No complete triangles in cap band")
    areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    area_sum = float(areas.sum())
    if not np.isfinite(area_sum) or area_sum <= 0:
        raise ValueError("Cap triangle area is invalid")
    centroid = (triangles.mean(axis=1) * areas[:, None]).sum(axis=0) / area_sum
    return centroid, int(len(triangles)), area_sum, extreme


def verify_identity_visuals(urdf: Path, links: set[str]) -> None:
    """Fail rather than silently misinterpret a mesh with a visual transform."""
    root = ET.parse(urdf).getroot()
    for link_name in links:
        link = next((node for node in root.findall("link") if node.get("name") == link_name), None)
        if link is None:
            raise ValueError(f"Missing link in URDF: {link_name}")
        visual = link.find("visual")
        if visual is None or visual.find("geometry/mesh") is None:
            raise ValueError(f"Missing visual mesh for link: {link_name}")
        origin = visual.find("origin")
        xyz = [float(v) for v in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()]
        rpy = [float(v) for v in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()]
        scale_node = visual.find("geometry/mesh")
        scale = [float(v) for v in scale_node.get("scale", "1 1 1").split()]
        if xyz != [0.0, 0.0, 0.0] or rpy != [0.0, 0.0, 0.0] or scale != [1.0, 1.0, 1.0]:
            raise ValueError(
                f"Non-identity visual transform for {link_name}: xyz={xyz}, rpy={rpy}, scale={scale}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tip-band-mm", type=float, default=0.5)
    parser.add_argument("--wrist-band-mm", type=float, default=0.1)
    args = parser.parse_args()

    urdf = args.urdf.resolve()
    root_dir = urdf.parent
    robot = HandURDF(urdf)
    selected_links = {"hand_base_link", *TIP_LINKS.values()}
    verify_identity_visuals(urdf, selected_links)
    transforms, _, _ = robot.fk({})

    tip_band_m = args.tip_band_mm / 1000.0
    wrist_band_m = args.wrist_band_mm / 1000.0
    mesh_files = {
        "wrist_ref": ("hand_base_link", root_dir / "meshes" / "hand_base_link.STL"),
        **{
            name: (link, root_dir / "meshes" / f"{link}.STL")
            for name, link in TIP_LINKS.items()
        },
    }

    points = {}
    audit = {}
    for point_name, (link, mesh_path) in mesh_files.items():
        mesh = trimesh.load_mesh(mesh_path, process=True)
        vertices = np.asarray(mesh.vertices, dtype=float)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if point_name == "wrist_ref":
            xyz, face_count, cap_area, extreme = area_weighted_cap_centroid(
                vertices, faces, axis=2, side="min", band_m=wrist_band_m
            )
            rule = "area_weighted_centroid_of_bottom_z_cap"
            band_mm = args.wrist_band_mm
        else:
            xyz, face_count, cap_area, extreme = area_weighted_cap_centroid(
                vertices, faces, axis=2, side="max", band_m=tip_band_m
            )
            rule = "area_weighted_centroid_of_distal_plus_z_cap"
            band_mm = args.tip_band_mm

        root_xyz = transforms[link] @ np.r_[xyz, 1.0]
        points[point_name] = {
            "link": link,
            "local_xyz_m": [float(v) for v in xyz],
            "zero_pose_root_xyz_m": [float(v) for v in root_xyz[:3]],
            "selection_rule": rule,
            "selection_axis": "local_z",
            "selection_side": "min" if point_name == "wrist_ref" else "max",
            "cap_band_mm": float(band_mm),
        }
        audit[link] = {
            "mesh": f"meshes/{link}.STL",
            "sha256": sha256(mesh_path),
            "vertices_after_process": int(len(vertices)),
            "faces": int(len(faces)),
            "bounds_m": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
            "cap_extreme_m": extreme,
            "cap_triangle_count": face_count,
            "cap_area_m2": cap_area,
        }

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "version": "v4_mesh_reference_points",
        "status": "initial_simulation_geometry; not_hardware_calibrated",
        "purpose": "Initial fixed points for MANO-R geometric fitting in simulation",
        "coordinate_convention": {
            "urdf_root": robot.root,
            "units": "m",
            "angles": "rad",
            "local_xyz_frame": "the link frame named in each point record",
            "root_xyz_frame": "hand_base_link at URDF zero pose",
            "tip_semantics": "distal-end cap center; not a measured contact-patch center",
            "wrist_semantics": "bottom installation/palm reference center; not asserted anatomical wrist",
        },
        "source": {
            "urdf": str(urdf),
            "urdf_sha256": sha256(urdf),
            "visual_origin_assumption": "all selected visual origins are identity and no mesh scale is specified",
        },
        "points": points,
        "mesh_audit": audit,
        "limitations": [
            "STL contains geometry but no semantic wrist or fingertip labels.",
            "The distal-cap convention is for the zero-pose simulation model.",
            "Real assembly zero, coating, compliance, and contact location are not calibrated.",
            "This point set is not a measured MANO shape or a retargeting result.",
        ],
    }
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    for name, value in points.items():
        xyz_mm = np.asarray(value["local_xyz_m"]) * 1000.0
        root_mm = np.asarray(value["zero_pose_root_xyz_m"]) * 1000.0
        print(f"{name:12s} local_mm={np.round(xyz_mm, 3)} root_zero_mm={np.round(root_mm, 3)}")


if __name__ == "__main__":
    main()
