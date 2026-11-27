"""research report: a report is built again from what is kept, never stored.

The report folder (in git: a problem's reports/<name>/, or an experiment's report/) holds report.json and the author's parts (the format of
autoresearch/reporter). Its report.json may add "draw": [{"run", "frame", "view", "as"}]: pictures drawn from the run's STORED predictions,
its raw data and ground truth (research/draw.py, no network runs); "as" is the name the parts use (e.g. "media/e1a.png"). The build goes to
<runs>/<problem>/reports/<name>/ (report.html, media/), outside git; the paths of values and sources are made absolute on the way."""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
from typing import Any, Dict

from research.launch import load_problem
from research.paths import CORE, ResearchError, child_env, interpreter, module_path, problem_dir, read_json, repo, runs_root, write_json

REPORTER = os.path.join(CORE, "reporter", "reporter.py")


def _abs(base: str, value: Any) -> Any:
    return os.path.normpath(os.path.join(base, value)) if isinstance(value, str) and not os.path.isabs(value) else value


def build(problem: str, folder: str, strict: bool = False) -> Dict[str, Any]:
    src = os.path.abspath(folder)
    if not os.path.isfile(os.path.join(src, "report.json")):
        raise ResearchError(f"{src} holds no report.json (a report folder: report.json and parts/, autoresearch/reporter)")
    spec = read_json(os.path.join(src, "report.json"))
    p = load_problem(problem)
    name = os.path.basename(src.rstrip("/")) if os.path.basename(src) != "report" else os.path.basename(os.path.dirname(src))
    out = os.path.join(runs_root(problem), "reports", name)
    if os.path.exists(out):
        shutil.rmtree(out)
    shutil.copytree(src, out)
    for v in spec.get("values", {}).values():
        if "file" in v:
            v["file"] = _abs(src, v["file"])
    spec["sources"] = [_abs(src, s) for s in spec.get("sources", [])]
    for s in spec.get("sections", []):                      # an svg is named relative to parts/ (reporter): one outside the report folder
        svg = s.get("svg")                                  # would not be found from the copy, so it is made absolute
        if svg and not os.path.isabs(svg):
            full = os.path.normpath(os.path.join(src, "parts", svg))
            if os.path.relpath(full, src).startswith(".."):
                s["svg"] = full
    draw = [{"run": d["run"], "frame": d["frame"], "view": d.get("view", "overview"), "out": os.path.join(out, d["as"])} for d in spec.pop("draw", [])]
    env = child_env(RESEARCH_REPO=repo(), RESEARCH_RUNS=runs_root(problem), PYTHONPATH=module_path(repo(), problem))
    if draw:
        r = subprocess.run([interpreter(p["python"]), "-m", "research.draw", problem], cwd=problem_dir(problem), env=env,
                           input=json.dumps(draw).encode(), capture_output=True)
        if r.returncode != 0:
            raise ResearchError(f"drawing the pictures failed: {r.stderr.decode('utf-8', 'replace')[-1500:]}")
    write_json(os.path.join(out, "report.json"), spec)
    r = subprocess.run([interpreter(p["python"]), REPORTER, "build", out] + (["--strict"] if strict else []), capture_output=True, env=env)
    if r.returncode != 0:
        raise ResearchError(f"the report build failed: {(r.stdout + r.stderr).decode('utf-8', 'replace')[-1500:]}")
    return {"report": os.path.join(out, "report.html"), "pictures": len(draw), "media": len(glob.glob(os.path.join(out, "media", "*")))}
