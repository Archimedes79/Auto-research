"""research report: a report is built again from what is kept, never stored.

The report folder (in git: a problem's reports/<name>/, or an experiment's report/) holds report.json and the author's parts (the format of
autoresearch/reporter). Its report.json may add "draw": [{"run", "frame", "view", "as"}]: pictures drawn from the run's STORED predictions,
its raw data and ground truth (research/draw.py, no network runs); "as" is the name the parts use (e.g. "media/e1a.png"). The build is a copy of
the report folder in <runs>/<problem>/reports/<name>/ (outside git) with the pictures drawn into it, report.html and media/; the copy's
report.json names the folder it came from ("origin"), where the reporter reads what lies outside the report folder."""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
from typing import Any, Dict

from research.launch import load_problem
from research.paths import CORE, ResearchError, child_env, core_step, interpreter, module_path, problem_dir, read_json, repo, runs_root, write_json

REPORTER = os.path.join(CORE, "reporter", "reporter.py")


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
    draw = [{"run": d["run"], "frame": d["frame"], "view": d.get("view", "overview"), "out": os.path.join(out, d["as"])} for d in spec.pop("draw", [])]
    env = child_env(RESEARCH_REPO=repo(), RESEARCH_RUNS=runs_root(problem), PYTHONPATH=module_path(repo(), problem))
    if draw:
        r = subprocess.run(core_step(p["python"], "research.draw", problem), cwd=problem_dir(problem), env=env,
                           input=json.dumps(draw).encode(), capture_output=True)
        if r.returncode != 0:
            raise ResearchError(f"drawing the pictures failed: {r.stderr.decode('utf-8', 'replace')[-1500:]}")
    write_json(os.path.join(out, "report.json"), dict(spec, origin=src))
    r = subprocess.run([interpreter(p["python"]), REPORTER, "build", out] + (["--strict"] if strict else []), capture_output=True, env=env)
    if r.returncode != 0:
        raise ResearchError(f"the report build failed: {(r.stdout + r.stderr).decode('utf-8', 'replace')[-1500:]}")
    said = r.stdout.decode("utf-8", "replace").splitlines()
    warnings = [ln.strip() for ln in said if "no architecture drawing" in ln or (" PENDING mark" in ln and " 0 PENDING mark" not in ln)]
    return {"report": os.path.join(out, "report.html"), "pictures": len(draw), "media": len(glob.glob(os.path.join(out, "media", "*"))),
            **({"warnings": warnings} if warnings else {})}
