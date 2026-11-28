"""A small html writer for the report page of an evaluation: headings and tables. No styling dependencies; the page opens from the file system."""
from __future__ import annotations

import html
import os
from typing import Any, Iterable, Sequence

CSS = """body{font-family:Segoe UI,Arial,sans-serif;margin:18px;max-width:1500px}h1{font-size:20px}h2{font-size:16px;margin-top:26px}
table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #bbb;padding:3px 7px;text-align:right}th{background:#eee}
td.l,th.l{text-align:left}.warn{background:#fff3cd}.fail{background:#f8d7da}.ok{background:#e8f5e9}"""


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
