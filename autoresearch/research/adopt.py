"""adopt: a winning arm becomes the problem's champion, on the owner's decided decision.

1. the arm's verdict is KEEP (or the adoption is by hand, with a reason);
2. the experiment's change of the problem's code (its snapshot against the commit it started from, the experiments/ folder left out) is applied
   to the working tree three-way, and the arm's settings become the defaults of the base config;
3. the anchor test: the arm's runs are evaluated again with that code, into a fresh folder, and must give the numbers of their records;
4. champions/<problem>/ records the champion (the old one stays as a comparison network); one commit of code, base config and champion facts.
A change of the core (autoresearch/) inside an experiment is not adopted here: core code is reviewed and committed on its own.

register: the same facts for a network measured by an experiment of kind `measure` (an imported or re-measured network), without a code change."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional

from research import gates, gitio
from research.launch import load_card, load_problem
from research.paths import ResearchError, experiment_dir, interpreter, problem_dir, read_json, record_dir, rel, repo, write_json


def champions_dir(problem: str) -> str:
    return os.path.join(repo(), "champions", problem)


def _facts(problem: str, exp_id: str, arm: str, card: Dict[str, Any], decision: str, how: str) -> Dict[str, Any]:
    seeds = {}
    for s in card["seeds"]:
        rid = f"{exp_id}.{arm}_s{s}"
        rec = record_dir(problem, rid)
        run, res = read_json(os.path.join(rec, "run.json")), read_json(os.path.join(rec, "result.json"))
        seeds[str(s)] = {"run": rid, "config": read_json(os.path.join(rec, "config.json")), "weights": {"run_dir": f"{{runs}}/{rid}",
                         "checkpoint": res["identity"].get("checkpoint"), "sha256_16": res["identity"].get("checkpoint_hash")},
                         "split": res["split"], "frames": res["frames"], "scenes": res["scenes"], "numbers": res["numbers"],
                         "per_scene": res["per_scene"], "cost": res.get("cost"), "identity": res["identity"], "commit": run["commit"]}
    prefix = load_problem(problem)["tag_prefix"]
    return {"name": f"{exp_id}.{arm}", "problem": problem, "experiment": exp_id, "arm": arm, "tag": f"{prefix}/{exp_id}",
            "decision": decision, "how": how, "date": time.strftime("%Y-%m-%d %H:%M"), "seeds": seeds}


def _record(problem: str, facts: Dict[str, Any], role: str) -> List[str]:
    """write champions/<problem>/<name>/network.json and update champions.json; -> the paths written"""
    base = champions_dir(problem)
    folder = os.path.join(base, facts["name"])
    write_json(os.path.join(folder, "network.json"), facts)
    drawing = os.path.join(problem_dir(problem), "training", "architecture.svg")
    if os.path.isfile(drawing):
        shutil.copyfile(drawing, os.path.join(folder, "architecture.svg"))
    idx_path = os.path.join(base, "champions.json")
    try:
        idx = read_json(idx_path)
    except (OSError, ValueError):
        idx = {"champion": None, "comparison": [], "history": []}
    if role == "champion":
        if idx.get("champion") and idx["champion"] != facts["name"]:
            idx["comparison"] = [idx["champion"]] + [c for c in idx["comparison"] if c != idx["champion"]]
        idx["champion"] = facts["name"]
    elif facts["name"] not in idx["comparison"]:
        idx["comparison"].append(facts["name"])
    idx["history"].append({"name": facts["name"], "role": role, "decision": facts["decision"], "date": facts["date"]})
    write_json(idx_path, idx)
    return [folder, idx_path]


def _anchor(problem: str, rids: List[str], p: Dict[str, Any], tolerance: float) -> None:
    env = dict(os.environ, RESEARCH_REPO=repo(), PYTHONPATH=os.pathsep.join([problem_dir(problem), os.path.join(repo(), "autoresearch"), repo()]))
    for rid in rids:
        r = subprocess.run([interpreter(p["python"]), "-m", "research.anchor", problem, rid, "--tolerance", str(tolerance)], cwd=problem_dir(problem),
                           env=env, capture_output=True)
        if r.returncode != 0:
            raise ResearchError(f"the anchor test of {rid} failed with the adopted code:\n{r.stdout.decode()[-1500:]}{r.stderr.decode('utf-8', 'replace')[-1500:]}")


def adopt(problem: str, exp_id: str, arm: str, decision: str, by_hand: str = "", tolerance: float = 0.0, commit: bool = True) -> Dict[str, Any]:
    if decision not in gates.decided(problem):
        raise ResearchError(f"{decision} is not a decided decision of {problem}: adopting a champion is the owner's decision")
    root = repo()
    p = load_problem(problem)
    tag = f"{p['tag_prefix']}/{exp_id}"
    snap = gitio.tag_commit(root, tag)
    card = load_card(problem, exp_id, at=snap)
    verdict = read_json(os.path.join(experiment_dir(problem, exp_id), "verdict.json"))
    v = verdict.get("arms", {}).get(arm, {}).get("verdict")
    if v != "KEEP" and not by_hand:
        raise ResearchError(f"the verdict of {exp_id}.{arm} is {v}: only a KEEP is adopted (or by hand, with a reason)")
    base = gitio.git(root, "rev-parse", f"{snap}^")
    pdir = rel(problem_dir(problem))
    core = [f for f in gitio.changed_files(root, base, snap, ["autoresearch"])]
    if core:
        raise ResearchError(f"{exp_id} changed core code ({', '.join(core[:5])}): commit and review core code on its own, then adopt")
    paths = [pdir, f":(exclude){pdir}/experiments", f":(exclude){pdir}/ledger.csv"]
    changed = gitio.changed_files(root, base, snap, paths)
    patch = gitio.diff(root, base, snap, paths)
    gitio.apply(root, patch)
    cfg_path = os.path.join(problem_dir(problem), p["config"])
    cfg = read_json(cfg_path)
    cfg.update(card["arms"][arm].get("set", {}))
    write_json(cfg_path, cfg)
    _anchor(problem, [f"{exp_id}.{arm}_s{s}" for s in card["seeds"]], p, tolerance)
    facts = _facts(problem, exp_id, arm, card, decision, f"by hand: {by_hand}" if by_hand else "verdict KEEP")
    written = _record(problem, facts, "champion")
    paths_c = sorted(set(changed + [rel(cfg_path)] + [rel(x) for x in written]))
    c = gitio.commit_paths(root, paths_c, f"adopt {exp_id}.{arm} ({problem}) as champion under {decision}\n\n{facts['how']}; code of {tag}, anchor test passed") if commit else ""
    return {"champion": facts["name"], "changed": changed, "commit": c}


def register(problem: str, exp_id: str, arm: str, decision: str, role: str, commit: bool = True) -> Dict[str, Any]:
    """the facts of a measured network (an experiment of kind measure) as champion or comparison network, on the owner's decision"""
    if role not in ("champion", "comparison"):
        raise ResearchError("role: champion or comparison")
    if decision not in gates.decided(problem):
        raise ResearchError(f"{decision} is not a decided decision of {problem}")
    p = load_problem(problem)
    snap = gitio.tag_commit(repo(), f"{p['tag_prefix']}/{exp_id}")
    card = load_card(problem, exp_id, at=snap)
    if card.get("kind") != "measure":
        raise ResearchError(f"{exp_id} is not a measurement: a compared arm becomes champion through adopt")
    facts = _facts(problem, exp_id, arm, card, decision, f"registered as {role}")
    written = _record(problem, facts, role)
    c = gitio.commit_paths(repo(), [rel(x) for x in written], f"register {exp_id}.{arm} ({problem}) as {role} under {decision}") if commit else ""
    return {"name": facts["name"], "role": role, "commit": c}
