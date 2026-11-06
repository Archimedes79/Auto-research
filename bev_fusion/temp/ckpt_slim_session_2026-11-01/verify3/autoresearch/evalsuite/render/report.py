"""A small html writer for the index pages: headings, tables, thumbnails and links with paths relative to the page.
No styling dependencies; the pages open from the file system."""
from __future__ import annotations

import html
import os
from typing import Any, Iterable, List, Sequence

CSS = """body{font-family:Segoe UI,Arial,sans-serif;margin:18px;max-width:1500px}h1{font-size:20px}h2{font-size:16px;margin-top:26px}
table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #bbb;padding:3px 7px;text-align:right}th{background:#eee}
td.l,th.l{text-align:left}.warn{background:#fff3cd}.fail{background:#f8d7da}.ok{background:#e8f5e9}code{background:#f3f3f3;padding:1px 3px}
img{border:1px solid #ccc;margin:2px}.note{color:#555;font-size:12px}"""


def rel(path: str, start: str) -> str:
    return os.path.relpath(path, start).replace(os.sep, "/")


def fmt(v: Any) -> str:
    if isinstance(v, float):
        return "nan" if v != v else f"{v:.4g}"
    return html.escape(str(v))


def table(rows: Sequence[Sequence[Any]], left_cols: Sequence[int] = (0,), status_col: int = -1) -> str:
    """rows[0] is the header. status_col: a cell 'ok' | 'warn' | 'fail' colours its row."""
    out = ["<table>"]
    for i, r in enumerate(rows):
        tag = "th" if i == 0 else "td"
        cls = ""
        if i > 0 and 0 <= status_col < len(r) and str(r[status_col]) in ("ok", "warn", "fail"):
            cls = f' class="{r[status_col]}"'
        cells = "".join(f'<{tag}{" class=l" if j in left_cols else ""}>{fmt(c)}</{tag}>' for j, c in enumerate(r))
        out.append(f"<tr{cls}>{cells}</tr>")
    out.append("</table>")
    return "\n".join(out)


def image(path: str, start: str, width: int = 420, caption: str = "") -> str:
    r = rel(path, start)
    cap = f"<div class=note>{html.escape(caption)}</div>" if caption else ""
    return f'<div style="display:inline-block;vertical-align:top"><a href="{r}"><img src="{r}" width="{width}"></a>{cap}</div>'


def link(path: str, start: str, text: str) -> str:
    return f'<a href="{rel(path, start)}">{html.escape(text)}</a>'


def write(path: str, title: str, sections: Iterable[tuple]) -> str:
    """sections: (heading, html body) pairs -> the page at `path`"""
    parts = [f"<!doctype html><meta charset=utf-8><title>{html.escape(title)}</title><style>{CSS}</style>",
             f"<h1>{html.escape(title)}</h1>"]
    for head, body in sections:
        if head:
            parts.append(f"<h2>{html.escape(head)}</h2>")
        parts.append(body)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    return path


def links_in(path: str) -> List[str]:
    """all href / src targets of a page (for tests: every one must exist)"""
    import re
    with open(path, encoding="utf-8") as f:
        txt = f.read()
    return [m for m in re.findall(r'(?:href|src)="([^"#]+)"', txt) if not m.startswith(("http:", "https:"))]
