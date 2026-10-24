"""Measure the operator's interphalangeal angle ratios from the fitted sequence.

The L20 URDF hard-codes DIP = 0.8917 * PIP and thumb IP = 1.1619 * MCP.  Whether
that is a good approximation depends on the operator's own joint ratio.  This
script estimates the operator's DIP/PIP and IP/MCP ratios geometrically from the
21 fitted landmarks (angle between consecutive bone directions) so the coupling
method can be designed against measured numbers instead of a textbook value.

Caveat: these angles come from vector directions only.  They include any axial
twist contamination in the v21/v23 fit and are not anatomical joint angles from
a real goniometer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from run_r_l20_feature_baseline_v15 import POINT_ORDER

FINGERS = ["index", "middle", "ring", "pinky"]
PIP_OPTS = {"min_pip_rad": 0.2}


def unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-12 else vector


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    cosine = float(np.clip(np.dot(unit(a), unit(b)), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence", type=Path,
        default=Path("experiment/v23_pinky_distal_twist/run2_personal_mapping_test/"
                     "fixed_r/r_sequence_v22.json"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("experiment/v24_stage1_R_L20_audit/human_joint_ratio.json"))
    args = parser.parse_args()

    frames = json.loads(args.sequence.read_text(encoding="utf-8"))["frames"]
    index = {name: i for i, name in enumerate(POINT_ORDER)}
    points = np.asarray([f["source_reference_landmarks_m"] for f in frames], dtype=np.float64)

    samples = {finger: {"pip": [], "dip": [], "ratio": [], "share": []} for finger in FINGERS}
    thumb = {"mcp": [], "ip": [], "ratio": []}

    for frame in points:
        for finger in FINGERS:
            mcp, pip, dip, tip = (frame[index[f"{finger}_{n}"]] for n in ("mcp", "pip", "dip", "tip"))
            pip_angle = angle_between(pip - mcp, dip - pip)
            dip_angle = angle_between(dip - pip, tip - dip)
            if pip_angle >= np.degrees(PIP_OPTS["min_pip_rad"]):
                samples[finger]["pip"].append(pip_angle)
                samples[finger]["dip"].append(dip_angle)
                samples[finger]["ratio"].append(dip_angle / pip_angle)
                samples[finger]["share"].append(dip_angle / (pip_angle + dip_angle))
        cmc, mcp, ip, tip = (frame[index[f"thumb_{n}"]] for n in ("cmc", "mcp", "ip", "tip"))
        mcp_angle = angle_between(mcp - cmc, ip - mcp)
        ip_angle = angle_between(ip - mcp, tip - ip)
        if mcp_angle >= np.degrees(PIP_OPTS["min_pip_rad"]):
            thumb["mcp"].append(mcp_angle)
            thumb["ip"].append(ip_angle)
            thumb["ratio"].append(ip_angle / mcp_angle)

    def summary(values):
        array = np.asarray(values)
        return {
            "count": int(array.size),
            "mean_deg": float(array.mean()),
            "median_deg": float(np.median(array)),
            "p05_deg": float(np.percentile(array, 5)),
            "p95_deg": float(np.percentile(array, 95)),
        }

    result = {
        "version": "v24_human_joint_ratio",
        "sequence": str(args.sequence),
        "frame_count": len(frames),
        "min_pip_deg": np.degrees(PIP_OPTS["min_pip_rad"]),
        "definition": "angle between consecutive bone directions; not a goniometer reading",
        "fingers": {},
        "thumb": {},
        "l20_mimic": {"dip_over_pip": 0.8917, "thumb_ip_over_mcp": 1.1619},
    }
    for finger in FINGERS:
        data = samples[finger]
        result["fingers"][finger] = {
            "pip": summary(data["pip"]),
            "dip": summary(data["dip"]),
            "dip_over_pip": {
                "mean": float(np.mean(data["ratio"])),
                "median": float(np.median(data["ratio"])),
                "l20_ratio": 0.8917,
                "l20_minus_human_median": float(0.8917 - np.median(data["ratio"])),
            },
            "dip_share_of_interphalangeal_curl": {
                "median": float(np.median(data["share"])),
                "mean": float(np.mean(data["share"])),
                "l20_share": 0.8917 / (1.0 + 0.8917),
            },
        }
    result["thumb"] = {
        "mcp": summary(thumb["mcp"]),
        "ip": summary(thumb["ip"]),
        "ip_over_mcp": {
            "mean": float(np.mean(thumb["ratio"])),
            "median": float(np.median(thumb["ratio"])),
            "l20_ratio": 1.1619,
            "l20_minus_human_median": float(1.1619 - np.median(thumb["ratio"])),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "status": "done",
        "ratios": {f: round(result["fingers"][f]["dip_over_pip"]["median"], 3) for f in FINGERS},
        "thumb_ratio": round(result["thumb"]["ip_over_mcp"]["median"], 3),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
