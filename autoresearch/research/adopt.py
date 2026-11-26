"""adopt: a winning arm becomes the problem's champion, on the owner's decided decision.

1. the arm's verdict is KEEP (or the adoption is by hand, with a reason);
2. the arm's settings become the defaults of the base config (its code is committed already: `start` refuses uncommitted code, and a new option
   enters the code default-off);
3. the anchor test: the arm's runs are evaluated again with the working tree's code, into a fresh folder, and must give the numbers of their
   records (within problem.json evaluate.anchor_tolerance); a failure leaves the base config as it was;
4. champions/<problem>/ records the champion (the old one stays as a comparison network); one commit of base config and champion facts.

register: the same facts for a network measured by an experiment of kind `measure` (an imported or re-measured network), without a code change."""
from __future__ import annotations

import json
import os
import shutil
import time
from typing import Any, Dict, List, Optional

from research import gates, gitio, spec
from research.launch import load_card, load_problem
from research.paths import ResearchError, experiment_dir, problem_dir, read_json, record_dir, rel, repo, write_json


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
    from research import upkeep
    for rid in rids:
        res = upkeep.anchor(problem, rid, tolerance)
        if not res["agree"]:
            raise ResearchError(f"the anchor test of {rid} failed with the adopted settings: {json.dumps(res['differ'], default=str)[:1500]}")


def adopt(problem: str, exp_id: str, arm: str, decision: str, by_hand: str = "", tolerance: Optional[float] = None, commit: bool = True) -> Dict[str, Any]:
    if decision not in gates.decided(problem):
        raise ResearchError(f"{decision} is not a decided decision of {problem}: adopting a champion is the owner's decision")
    root = repo()
    p = load_problem(problem)
    tag = f"{p['tag_prefix']}/{exp_id}"
    snap = gitio.tag_commit(root, tag)
    if not snap:
        raise ResearchError(f"{exp_id} has no snapshot {tag}")
    card = dict(load_card(problem, exp_id, at=snap), seeds=load_card(problem, exp_id)["seeds"])
    verdict = read_json(os.path.join(experiment_dir(problem, exp_id), "verdict.json"))
    v = verdict.get("arms", {}).get(arm, {}).get("verdict")
    if v != "KEEP" and not by_hand:
        raise ResearchError(f"the verdict of {exp_id}.{arm} is {v}: only a KEEP is adopted (or by hand, with a reason)")
    tol = p["evaluate"].get("anchor_tolerance", 0.0) if tolerance is None else tolerance
    cfg_path = os.path.join(problem_dir(problem), p["config"])
    with open(cfg_path, "rb") as f:
        before = f.read()
    write_json(cfg_path, spec.merged(read_json(cfg_path), card["arms"][arm].get("set", {})))      # the code is committed (start checks it): the arm's settings become the defaults
    try:
        _anchor(problem, [f"{exp_id}.{arm}_s{s}" for s in card["seeds"]], p, tol)
    except ResearchError:
        with open(cfg_path, "wb") as f:                                         # a failed anchor test leaves the working tree as it was
            f.write(before)
        raise
    facts = _facts(problem, exp_id, arm, card, decision, f"by hand: {by_hand}" if by_hand else "verdict KEEP")
    written = _record(problem, facts, "champion")
    paths_c = sorted({rel(cfg_path)} | {rel(x) for x in written})
    c = gitio.commit_paths(root, paths_c, f"adopt {exp_id}.{arm} ({problem}) as champion under {decision}\n\n{facts['how']}; settings of {tag} made the defaults, anchor test passed") if commit else ""
    return {"champion": facts["name"], "settings": card["arms"][arm].get("set", {}), "commit": c}


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
