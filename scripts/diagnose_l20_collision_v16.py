"""Diagnose raw MuJoCo contacts for a v16 q sequence.

This is deliberately a diagnostic, not an optimization term.  The current
URDF-derived meshes may overlap at zero pose or across link boundaries, so the
script reports contacts and whether they can be explained by parent/child
adjacency instead of silently treating them as valid self-collision labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def related(parent: list[int], first: int, second: int) -> bool:
    return first == second or parent[first] == second or parent[second] == first


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--sequence-q", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import mujoco

    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    sequence = json.loads(args.sequence_q.resolve().read_text(encoding="utf-8"))
    parent = [int(value) for value in model.body_parentid]
    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}" for i in range(model.nbody)]
    frames = []
    for item in sequence["frames"]:
        data = mujoco.MjData(model)
        for name, value in item["q_full_rad"].items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                data.qpos[model.jnt_qposadr[jid]] = value
        mujoco.mj_forward(model, data)
        contacts = []
        non_adjacent = []
        for contact in data.contact[:data.ncon]:
            first_body = int(model.geom_bodyid[contact.geom1])
            second_body = int(model.geom_bodyid[contact.geom2])
            item_contact = {
                "body_pair": [body_names[first_body], body_names[second_body]],
                "body_ids": [first_body, second_body],
                "distance_m": float(contact.dist),
                "parent_child_related": related(parent, first_body, second_body),
            }
            contacts.append(item_contact)
            if not item_contact["parent_child_related"]:
                non_adjacent.append(item_contact)
        frames.append({
            "frame_id": item["frame_id"],
            "name": item["name"],
            "raw_contact_count": len(contacts),
            "non_adjacent_contact_count": len(non_adjacent),
            "minimum_contact_distance_m": float(min((c["distance_m"] for c in contacts), default=float("inf"))),
            "contacts": contacts,
            "non_adjacent_contacts": non_adjacent,
        })
    raw_counts = [item["raw_contact_count"] for item in frames]
    non_adj_counts = [item["non_adjacent_contact_count"] for item in frames]
    payload = {
        "version": "v16_L20_collision_diagnostic",
        "status": "completed_raw_contact_diagnostic_not_collision_metric",
        "scene": str(args.scene.resolve()),
        "sequence_q": str(args.sequence_q.resolve()),
        "summary": {
            "frame_count": len(frames),
            "max_raw_contact_count": max(raw_counts, default=0),
            "max_non_adjacent_contact_count": max(non_adj_counts, default=0),
            "frames_with_non_adjacent_contacts": int(sum(count > 0 for count in non_adj_counts)),
            "interpretation": "current URDF collision meshes are not yet a cleaned contact descriptor",
        },
        "frames": frames,
        "limitations": [
            "The scene is derived directly from the supplied URDF meshes; contact overlap may reflect CAD/link overlap rather than a meaningful collision.",
            "No collision penalty was added to v16 q optimization.",
            "This diagnostic does not evaluate contact normals, object interaction or grasp success.",
        ],
    }
    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "summary": payload["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
