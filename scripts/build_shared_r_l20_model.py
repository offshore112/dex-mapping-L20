"""Build a shared target R_L20 model from the L20 URDF/meshes and MANO-R fit."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from l20_kinematics import HandURDF  # noqa: E402
from shared_r_l20 import SharedRL20  # noqa: E402


def file_record(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def export_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    lines = ["# Shared R_L20 MANO-R reference surface; metres"]
    lines.extend("v " + " ".join(f"{float(x):.10f}" for x in row) for row in vertices)
    lines.extend("f " + " ".join(str(int(x) + 1) for x in row) for row in faces)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def mesh_records(urdf_path: Path) -> list[dict]:
    root = ET.parse(urdf_path).getroot()
    records = []
    seen = set()
    for link in root.findall("link"):
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                continue
            filename = mesh.attrib["filename"]
            path = (urdf_path.parent / filename).resolve()
            if filename in seen:
                continue
            seen.add(filename)
            if not path.is_file():
                raise FileNotFoundError(path)
            origin = visual.find("origin")
            records.append({
                "filename": filename,
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
                "scale": [float(x) for x in mesh.attrib.get("scale", "1 1 1").split()],
                "visual_origin_xyz": [float(x) for x in (origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()],
                "visual_origin_rpy": [float(x) for x in (origin.attrib.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()],
                "links_using_mesh": [link.attrib["name"]],
            })
    return records


def joint_records(robot: HandURDF) -> list[dict]:
    return [
        {
            "name": joint.name,
            "type": joint.kind,
            "parent": joint.parent,
            "child": joint.child,
            "origin_xyz_m": joint.origin[:3, 3].tolist(),
            "origin_rotation": joint.origin[:3, :3].tolist(),
            "axis": joint.axis.tolist(),
            "lower_rad": joint.lower,
            "upper_rad": joint.upper,
            "mimic": joint.mimic,
        }
        for joint in robot.joints
    ]


def build_point_sources(target_path: Path, tips_path: Path) -> tuple[list[str], list[dict], np.ndarray]:
    target = json.loads(target_path.read_text(encoding="utf-8"))
    tips = json.loads(tips_path.read_text(encoding="utf-8"))
    entries = target["points"]
    point_order = [entry["name"] for entry in entries]
    target_points = np.asarray([entry["xyz_m"] for entry in entries], dtype=float)
    sources = []
    for entry in entries:
        name = entry["name"]
        if name == "wrist":
            local = tips["points"]["wrist_ref"]["local_xyz_m"]
            sources.append({"name": name, "kind": "root_local", "local_xyz_m": local})
        elif "joint" in entry:
            sources.append({"name": name, "kind": "joint_frame", "joint": entry["joint"]})
        elif "link" in entry:
            local = tips["points"][name]["local_xyz_m"]
            sources.append({"name": name, "kind": "link_local", "link": entry["link"], "local_xyz_m": local})
        else:
            raise ValueError(f"No kinematic source for target point {name}")
    return point_order, sources, target_points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--tips", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", default="v10", help="artifact version label, e.g. v10 or v11")
    args = parser.parse_args()
    version = args.version

    urdf_path = args.urdf.resolve()
    fit_path = args.fit.resolve()
    target_path = args.target.resolve()
    tips_path = args.tips.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    robot = HandURDF(urdf_path)
    fit = np.load(fit_path, allow_pickle=False)
    required = {"beta", "scale", "rotation", "translation_m", "vertices_m", "faces", "fitted_points_m", "target_points_m", "point_error_m"}
    missing = required - set(fit.files)
    if missing:
        raise ValueError(f"MANO-R fit is missing arrays: {sorted(missing)}")
    theta_ref = np.asarray(fit["theta_ref_hand_pose"], dtype=float) if "theta_ref_hand_pose" in fit.files else np.zeros(45, dtype=float)
    point_order, point_sources, target_points_from_json = build_point_sources(target_path, tips_path)
    if fit["target_points_m"].shape != target_points_from_json.shape or not np.allclose(fit["target_points_m"], target_points_from_json, atol=1e-10):
        raise ValueError("Target points in fit NPZ and target JSON do not match")

    mesh_info = mesh_records(urdf_path)
    mimic = [record for record in joint_records(robot) if record["mimic"] is not None]
    q_order = list(robot.independent)
    array_name = "r_l20_model.npz"
    np.savez_compressed(
        out / array_name,
        mano_vertices_ref_m=np.asarray(fit["vertices_m"], dtype=np.float64),
        mano_faces=np.asarray(fit["faces"], dtype=np.int32),
        target_landmarks_m=np.asarray(fit["target_points_m"], dtype=np.float64),
        fitted_landmarks_m=np.asarray(fit["fitted_points_m"], dtype=np.float64),
        point_error_m=np.asarray(fit["point_error_m"], dtype=np.float64),
        beta_R=np.asarray(fit["beta"], dtype=np.float64),
        theta_ref_hand_pose=theta_ref.astype(np.float64),
        scale=np.asarray(fit["scale"], dtype=np.float64),
        rotation=np.asarray(fit["rotation"], dtype=np.float64),
        translation_m=np.asarray(fit["translation_m"], dtype=np.float64),
    )
    export_obj(out / "r_l20_surface.obj", fit["vertices_m"], fit["faces"])

    manifest = {
        "version": f"{version}_shared_R_L20_model",
        "status": "constructed_initial_shared_target_model",
        "model_role": "one shared target virtual hand for all operators; not a per-operator hand",
        "root_frame": "hand_base_link",
        "units": {"length": "m_convention_pending_physical_calibration", "angle": "rad"},
        "sources": {
            "urdf": str(urdf_path),
            "urdf_file": file_record(urdf_path),
            "mano_fit_npz": file_record(fit_path),
            "target_landmarks": file_record(target_path),
            "tip_reference_points": file_record(tips_path),
        },
        "geometry": {
            "representation": "fixed MANO-R reference surface; dynamic pose is a later interface",
            "surface_obj": "r_l20_surface.obj",
            "beta_R": np.asarray(fit["beta"], dtype=float).tolist(),
            "theta_ref_hand_pose": theta_ref.tolist(),
            "scale": float(np.asarray(fit["scale"])),
            "rotation": np.asarray(fit["rotation"], dtype=float).tolist(),
            "translation_m": np.asarray(fit["translation_m"], dtype=float).tolist(),
            "vertex_count": int(fit["vertices_m"].shape[0]),
            "face_count": int(fit["faces"].shape[0]),
            "static_fit_rmse_mm": float(np.sqrt(np.mean(np.asarray(fit["point_error_m"]) ** 2)) * 1000),
            "static_fit_max_error_mm": float(np.max(np.asarray(fit["point_error_m"])) * 1000),
        },
        "kinematics": {
            "robot_name": robot.name,
            "root": robot.root,
            "link_count": len(robot.links),
            "joint_count": len(robot.joints),
            "joint_order": [joint.name for joint in robot.joints],
            "independent_q_order": q_order,
            "independent_q_count": len(q_order),
            "mimic_count": len(mimic),
            "joints": joint_records(robot),
            "mimic_relations": mimic,
            "point_order": point_order,
            "point_sources": point_sources,
            "mesh_count": len(mesh_info),
            "meshes": mesh_info,
        },
        "arrays": {"npz": array_name},
        "limitations": [
            "The MANO-R surface is a fixed reference geometry; no camera/action sequence is fitted here.",
            "The 21 L20 points use joint-frame and distal-cap proxies, not anatomical or measured contact points.",
            "The URDF defines q-level kinematics and mimic relations, not verified actuator commands u.",
            "Real scale, assembly zero, contact, compliance and driver mapping remain unverified.",
        ],
    }
    manifest_path = out / "r_l20_model.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    model = SharedRL20(manifest_path)
    zero_points = model.landmarks()
    zero_residual = zero_points - model.target_landmarks_m
    mid_q = np.asarray([(model.robot.bounds[name][0] + model.robot.bounds[name][1]) / 2 for name in model.q_order])
    mid_points = model.landmarks(mid_q)
    full_mid = model.expand_q(mid_q)
    validation = {
        "version": f"{version}_shared_R_L20_model_validation",
        "status": "passed_structure_and_zero_pose_semantic_point_check" if np.max(np.abs(zero_residual)) < 1e-9 else "failed_zero_pose_semantic_point_check",
        "root": model.robot.root,
        "link_count": len(model.robot.links),
        "joint_count": len(model.robot.joints),
        "independent_q_count": len(model.q_order),
        "mimic_count": len(mimic),
        "mesh_count": len(mesh_info),
        "zero_pose_landmark_max_abs_error_m": float(np.max(np.abs(zero_residual))),
        "zero_pose_landmark_rmse_m": float(np.sqrt(np.mean(zero_residual ** 2))),
        "mid_limit_pose_finite": bool(np.isfinite(mid_points).all()),
        "mid_limit_full_joint_count": len(full_mid),
        "arrays_finite": bool(np.isfinite(model.vertices_m).all() and np.isfinite(model.target_landmarks_m).all()),
        "q_semantics": "independent joint state q; actuator command u not defined",
    }
    (out / "validation.json").write_text(json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    report = f"""# {version}：共享 R_L20 几何/运动学模型

## 模型角色

这是由 L20 URDF/meshes 和 MANO-R 拟合参数构成的**单个共享目标虚拟手**。不同操作者共享同一组 `beta_R`、尺度、刚体对齐和参考几何；动作状态以后由第一阶段输出，再进入 L20 机构映射。

## 已构建部分

- MANO-R 固定参考表面：`r_l20_surface.obj`，{fit['vertices_m'].shape[0]} 顶点、{fit['faces'].shape[0]} 三角面。
- L20 运动学骨架：{len(robot.links)} 个 link、{len(robot.joints)} 个 revolute joint、根为 `{robot.root}`。
- 独立关节状态 `q`：{len(q_order)} 个，顺序见 `r_l20_model.json`；5 个 mimic 关系保留在完整关节状态中。
- 21 个语义点：关节框架点、腕部根点和五个末节局部指尖点，均输出到 `hand_base_link` 坐标系。
- 共享模型数组：`r_l20_model.npz`；结构和零姿态语义点检查：`validation.json`。

## 验收

`validation.json` 检查了 URDF 树、独立 q 数量、mimic 数量、所有 mesh 存在性、零姿态 21 点重算和中间限位姿态有限性。零姿态点重算不是动态重定向性能指标。

## 当前边界

模型定义了目标几何和 q 级运动学接口，未定义真实执行器 `u`，未接摄像头，未运行动作重定向，也未进行真实尺度、装配零位、接触或柔顺标定。
"""
    (out / "README.md").write_text(report, encoding="utf-8")
    print(f"built shared R_L20 model: {manifest_path}")
    print(json.dumps(validation, ensure_ascii=False))


if __name__ == "__main__":
    main()
