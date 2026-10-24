"""Render a Markdown file with LaTeX math to a standalone, readable HTML page.

VSCode's built-in Markdown preview does not typeset ``$...$`` / ``$$...$$``.
This script protects math spans from the Markdown parser (otherwise
underscores inside ``\\theta_R`` would be read as emphasis), converts the rest
to HTML, restores the math, and wraps the result in a page that loads MathJax
from a CDN.

Usage:
    python scripts/render_markdown_math_html.py docs/某文档.md -o docs/某文档.html
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import markdown

DISPLAY_MATH = re.compile(r"\$\$(.+?)\$\$", re.S)
INLINE_MATH = re.compile(r"(?<!\$)\$(?!\$)([^\n$]+?)\$(?!\$)")
TOKEN = "@@MATHPH{}@@"

TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<script>
window.MathJax = {{
  tex: {{
    inlineMath: [['$', '$']],
    displayMath: [['$$', '$$']],
    processEscapes: true
  }},
  options: {{ skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'] }}
}};
</script>
<script id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
  :root {{ color-scheme: light; }}
  body {{
    margin: 0 auto; max-width: 920px; padding: 48px 28px 96px;
    font-family: "Microsoft YaHei", "PingFang SC", "Segoe UI", system-ui, sans-serif;
    font-size: 16px; line-height: 1.75; color: #1f2328; background: #ffffff;
  }}
  h1 {{ font-size: 1.9em; border-bottom: 2px solid #d8dee4; padding-bottom: .3em; }}
  h2 {{ font-size: 1.45em; margin-top: 2.1em; border-bottom: 1px solid #e6e9ec; padding-bottom: .25em; }}
  h3 {{ font-size: 1.15em; margin-top: 1.6em; }}
  a {{ color: #0969da; }}
  code {{ background: #f2f4f6; padding: .15em .35em; border-radius: 4px; font-size: .92em; }}
  pre {{ background: #f6f8fa; padding: 14px 16px; border-radius: 6px; overflow-x: auto; }}
  pre code {{ background: none; padding: 0; }}
  table {{ border-collapse: collapse; margin: 1.1em 0; width: 100%; font-size: .95em; }}
  th, td {{ border: 1px solid #d8dee4; padding: 7px 11px; text-align: left; vertical-align: top; }}
  th {{ background: #f6f8fa; }}
  blockquote {{ margin: 1.1em 0; padding: .3em 1em; border-left: 4px solid #d0d7de; color: #57606a; background: #fafbfc; }}
  hr {{ border: none; border-top: 1px solid #e6e9ec; margin: 2.2em 0; }}
  mjx-container[display="true"] {{ margin: 1.1em 0 !important; overflow-x: auto; overflow-y: hidden; }}
  @media print {{
    body {{ max-width: none; padding: 0; }}
    pre, table, mjx-container {{ page-break-inside: avoid; }}
  }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def protect_math(text: str) -> tuple[str, list[str]]:
    """Replace math spans with inert tokens so Markdown cannot touch them."""
    store: list[str] = []

    def replace(match: re.Match) -> str:
        store.append(match.group(0))
        return TOKEN.format(len(store) - 1)

    text = DISPLAY_MATH.sub(replace, text)
    text = INLINE_MATH.sub(replace, text)
    return text, store


def restore_math(html: str, store: list[str]) -> str:
    for index, original in enumerate(store):
        html = html.replace(TOKEN.format(index), original)
    return html


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    text = args.source.read_text(encoding="utf-8")
    protected, store = protect_math(text)

    body = markdown.markdown(
        protected,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
        output_format="html5",
    )
    body = restore_math(body, store)

    leftover = re.findall(re.escape("@@MATHPH"), body)
    if leftover:
        raise SystemExit(f"{len(leftover)} math placeholders were not restored")

    title = args.title or text.splitlines()[0].lstrip("# ").strip()
    args.output.write_text(TEMPLATE.format(title=title, body=body), encoding="utf-8")
    print(f"wrote {args.output} ({len(store)} math spans, "
          f"{args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
