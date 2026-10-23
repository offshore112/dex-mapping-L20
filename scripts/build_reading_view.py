"""Build self-contained HTML views of the main document and linked records.

Usage: python scripts/build_reading_view.py
The Markdown package is available in the current local Python environment.
If using a fresh environment: python -m pip install Markdown
This script only renders documentation; it does not train or control robots.
"""

from html import escape, unescape
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit, urlunsplit

try:
    import markdown
    from markdown.extensions.toc import slugify_unicode
except ImportError as exc:
    raise SystemExit(
        "需要 Python Markdown 库：python -m pip install Markdown"
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
SOURCES = [ROOT / name for name in (
    "docs/研究思路与新手指南.md",
    "docs/新手执行清单.md",
    "docs/来源与核查.md",
    "docs/L20左手_URDF到虚拟人手R建模指南.md",
    "docs/MANO与L20独立环境使用说明.md",
    "dataset.md",
    "experiment/evaluation.md",
    "related_work/文献索引.md",
)]
RENDERED = {source.resolve() for source in SOURCES}

md = markdown.Markdown(
    extensions=["tables", "fenced_code", "toc", "sane_lists"],
    extension_configs={"toc": {"toc_depth": "2-3", "slugify": slugify_unicode}},
)
css = """
:root { --ink:#18283b; --muted:#5e6d7c; --blue:#235c91; --line:#dce4ea; }
* { box-sizing:border-box; }
html { scroll-behavior:smooth; scroll-padding-top:26px; }
body { margin:0; color:var(--ink); background:#f3f6f9;
 font-family:'Microsoft YaHei','PingFang SC',system-ui,sans-serif; line-height:1.85; }
.shell { max-width:1400px; margin:auto; display:grid; grid-template-columns:278px minmax(0,1fr); gap:28px; padding:32px; }
aside { position:sticky; top:24px; align-self:start; max-height:calc(100vh - 48px); overflow:auto; padding:22px;
 background:#edf3f8; border:1px solid var(--line); border-radius:14px; font-size:13px; }
.brand { font-size:22px; font-weight:750; color:var(--blue); }
.hint { color:var(--muted); font-size:12px; margin:8px 0 20px; }
aside ul { list-style:none; margin:0; padding-left:0; }
aside li { margin:9px 0; }
aside li ul { padding-left:13px; font-size:12px; }
a { color:var(--blue); text-decoration:none; overflow-wrap:anywhere; }
a:hover { text-decoration:underline; }
main { min-width:0; background:white; border:1px solid var(--line); border-radius:16px; padding:42px 48px; }
.eyebrow { color:var(--blue); letter-spacing:.12em; font-size:12px; font-weight:bold; }
.status { color:var(--muted); background:#f5f8fb; padding:12px 16px; border-radius:8px; font-size:13px; }
h1 { font-size:30px; line-height:1.45; margin:16px 0 20px; }
h2 { font-size:23px; line-height:1.5; margin-top:48px; padding-top:14px; border-top:2px solid var(--line); }
h3 { font-size:18px; margin-top:28px; }
p, li { overflow-wrap:anywhere; }
strong { color:#102a44; }
pre { padding:20px; background:#f0f5fa; border:1px solid var(--line); border-radius:10px;
 overflow:auto; line-height:1.7; font-size:13px; }
code { font-family:Consolas,'Microsoft YaHei',monospace; font-size:.88em; }
p code, li code { background:#eff4f8; border-radius:4px; padding:2px 5px; }
table { display:block; width:100%; overflow-x:auto; border-collapse:collapse; font-size:13px; line-height:1.7; margin:20px 0; }
td, th { border:1px solid var(--line); padding:11px 13px; min-width:95px; vertical-align:top; }
th { background:#eaf1f7; text-align:left; }
tbody tr:nth-child(even) { background:#fafbfd; }
blockquote { margin:22px 0; padding:12px 22px; border-left:4px solid #3c7bb1; background:#f3f7fb; }
footer { margin-top:40px; padding-top:20px; border-top:1px solid var(--line); color:var(--muted); font-size:12px; }
@media(max-width:900px) { .shell { display:block; padding:16px; } aside { position:static; max-height:260px; margin-bottom:18px; }
 main { padding:24px 20px; } h1 { font-size:25px; } h2 { font-size:21px; } }
@media print { body { background:white; font-size:11pt; } .shell { display:block; padding:0; } aside { display:none; }
 main { border:0; padding:0; } h2,h3 { break-after:avoid; } pre,blockquote,tr { break-inside:avoid; }
 table { display:table; font-size:9pt; } a { color:inherit; } }
"""

for source in SOURCES:
    source_text = source.read_text(encoding="utf-8")
    title = re.search(r"^# (.+)$", source_text, re.MULTILINE).group(1)
    md.reset()
    body = md.convert(source_text)

    def rewrite_local_link(match):
        raw = unescape(match.group(2))
        parts = urlsplit(raw)
        if not parts.scheme and not parts.netloc and parts.path.endswith(".md"):
            target = (source.parent / unquote(parts.path)).resolve()
            if target in RENDERED:
                raw = urlunsplit(parts._replace(path=parts.path[:-3] + ".html"))
        return match.group(1) + escape(raw, quote=True) + match.group(3)

    body = re.sub(r'(<a\b[^>]*href=")([^"]+)(")', rewrite_local_link, body)
    html = f"""<!doctype html>
<html lang="zh-CN">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)} · 新手研究指南</title><style>{css}</style></head>
<body><div class="shell">
<aside><div class="brand">dex-mapping</div>
<div class="hint">从研究想法到可验证的两阶段映射<br>2026-10-14 · 研究文档与实施记录</div>
<nav aria-label="文档目录">{md.toc}</nav></aside>
<main><div class="eyebrow">研究思路 · 新手指南</div>
<p class="status">当前阶段：独立环境与 MANO 加载核查已通过；L20 FK、R 拟合、基线与网络训练尚未运行。</p>
{body}
<footer>本页离线可读，无需加载外部资源。可编辑源文件：{escape(source.name)}。<br>
主文档与辅助记录在同一个项目文件夹中，已生成的阅读版之间可直接跳转。</footer>
</main></div></body></html>"""

    output = source.with_suffix(".html")
    output.write_text(html, encoding="utf-8")
    print(f"Created: {output.relative_to(ROOT)}")
