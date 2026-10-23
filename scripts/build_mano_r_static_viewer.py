"""Build a no-JavaScript MANO-R preview for browsers where Canvas fails."""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


CHAINS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12],
          [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]
COLORS = ["#d98243", "#3b82ac", "#569678", "#946cb0", "#c76c8a"]


def render(path, vertices, faces, target, fitted, elev, azim, title):
    fig, ax = plt.subplots(figsize=(7, 8), subplot_kw={"projection": "3d"})
    ax.add_collection3d(Poly3DCollection(vertices[faces] * 1000.0,
                                         facecolor="#3c83ad", edgecolor="#265873",
                                         linewidth=0.05, alpha=0.24))
    target_mm, fitted_mm = target * 1000.0, fitted * 1000.0
    ax.scatter(target_mm[:, 0], target_mm[:, 1], target_mm[:, 2], c="#d23c3c", s=20)
    ax.scatter(fitted_mm[:, 0], fitted_mm[:, 1], fitted_mm[:, 2], c="#2b9b62", s=15)
    for ids, color in zip(CHAINS, COLORS):
        ax.plot(target_mm[ids, 0], target_mm[ids, 1], target_mm[ids, 2], color=color, linewidth=1.5)
        ax.plot(fitted_mm[ids, 0], fitted_mm[ids, 1], fitted_mm[ids, 2], color=color,
                linewidth=1.1, linestyle="--")
    all_points = np.vstack((vertices, target, fitted)) * 1000.0
    lo, hi = all_points.min(0), all_points.max(0)
    center, span = (lo + hi) / 2.0, float(np.max(hi - lo))
    for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
        setter(c - span * 0.56, c + span * 0.56)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("root X (mm)"); ax.set_ylabel("root Y (mm)"); ax.set_zlabel("root Z (mm)")
    ax.set_title(title)
    ax.text2D(0.03, 0.96, "blue=MANO-R  red=L20 target  green=fitted points",
              transform=ax.transAxes, fontsize=8)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def data_uri(path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    d = np.load(args.fit.resolve())
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    vertices, faces = d["vertices_m"], d["faces"]
    target, fitted = d["target_points_m"], d["fitted_points_m"]
    views = [("front", 12, -72, "front"), ("side", 12, 18, "side"), ("top", 78, -72, "top")]
    files = []
    for title, elev, azim, name in views:
        p = out / f"mano_r_{name}.png"
        render(p, vertices, faces, target, fitted, elev, azim, f"MANO-R static fit - {title}")
        files.append((title, data_uri(p)))
    summary = json.loads((args.fit.resolve().parent / "fit_summary.json").read_text(encoding="utf-8"))
    cards = "".join(f'<figure><img src="{uri}" alt="{title}"><figcaption>{title}</figcaption></figure>'
                    for title, uri in files)
    html = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MANO-R 静态仿真预览</title><style>body{{margin:0;background:#edf2f6;color:#21374b;font-family:"Microsoft YaHei",system-ui,sans-serif}}header{{padding:22px 28px;background:#17344d;color:white}}h1{{margin:0 0 8px;font-size:22px}}p{{line-height:1.7}}main{{max-width:1500px;margin:auto;padding:18px}}.box{{background:white;border:1px solid #d5dfe7;border-radius:10px;padding:16px;margin-bottom:16px}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}figure{{margin:0;background:white;border:1px solid #d5dfe7;border-radius:10px;padding:10px}}img{{display:block;width:100%;height:auto}}figcaption{{text-align:center;padding:8px;font-weight:bold}}code{{font-family:Consolas,monospace}}@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}</style></head><body><header><h1>MANO-R 静态仿真预览</h1><p>本页面不使用 Canvas、不使用 JavaScript；蓝色为 MANO-R，红色为 L20 目标点，绿色为拟合点。</p></header><main><div class="box"><p><b>拟合 RMSE：</b>{summary['beta_scale_fit']['rmse_mm']:.2f} mm　<b>最大点误差：</b>{summary['beta_scale_fit']['max_mm']:.2f} mm</p><p>这是 URDF/CAD 零姿态的静态几何拟合，未加入真机、动力学或执行器约束。</p></div><div class="grid">{cards}</div></main></body></html>'''
    target = out / "mano_r_static_viewer.html"
    target.write_text(html, encoding="utf-8")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
