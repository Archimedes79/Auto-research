"""adopt: a winning arm becomes the problem's champion, on the owner's decided decision.

1. the arm's verdict is KEEP (or the adoption is by hand, with a reason);
2. the arm's settings become the defaults of the base config (its code is committed already: `start` refuses uncommitted code, and a new option
   enters the code default-off);
3. the anchor test: the arm's runs (their folders must still exist) are evaluated again with the working tree's code, into a fresh folder,
   and must give the numbers of their records (within problem.json evaluate.anchor_tolerance);
4. champions/<problem>/ records the champion (the old one stays as a comparison network); one commit of base config and champion facts.
Unless it gets through to that commit (a failed anchor test, an error, an interrupt), the base config and the champion files are put back
as they were.

register: the same facts for a network measured by an experiment of kind `measure` (an imported or re-measured network), without a code change;
its verdict must be DONE (`research judge --write`) and its run folders must exist."""
from __future__ import annotations

import json
import os
import shutil
import time
from typing import Any, Dict, List, Optional

from research import gates, gitio, spec
from research.launch import load_card, load_problem
from research.paths import ResearchError, experiment_dir, problem_dir, read_json, record_dir, rel, repo, run_dir, runs_root, write_json


def champions_dir(problem: str) -> str:
    return os.path.join(repo(), "champions", problem)


def _facts(problem: str, exp_id: str, arm: str, card: Dict[str, Any], decision: str, how: str) -> Dict[str, Any]:
    seeds = {}
    for s in card["seeds"]:
        rid = f"{exp_id}.{arm}_s{s}"
        rec = record_dir(problem, rid)
        try:
            run, res, cfg = (read_json(os.path.join(rec, f)) for f in ("run.json", "result.json", "config.json"))
            ident = res["identity"]
            seeds[str(s)] = {"run": rid, "config": cfg, "weights": {"run_dir": f"{{runs}}/{rid}", "checkpoint": ident.get("checkpoint"),
                                                                    "sha256_16": ident.get("checkpoint_hash") or ident.get("sha1")},     # bev names it sha1
                             "split": res["split"], "frames": res["frames"], "scenes": res["scenes"], "numbers": res["numbers"],
                             "per_scene": res["per_scene"], "cost": res.get("cost"), "identity": ident, "commit": run["commit"]}
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
            raise ResearchError(f"{rid}: its records cannot be read ({type(e).__name__}: {e}); only a finished, evaluated run becomes a champion") from None
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


def _run_folders(problem: str, rids: List[str]) -> None:
    """the seeds' run folders must exist: the facts name their weights, and the anchor test evaluates them"""
    gone = [r for r in rids if not os.path.isdir(run_dir(problem, r))]
    if gone:
        raise ResearchError(f"the run folders of {', '.join(gone)} are gone (in {runs_root(problem)}): their weights cannot be a champion")


def _saved(paths: List[str]) -> Dict[str, Optional[bytes]]:
    """the bytes of these files (None: absent), to put them back"""
    out: Dict[str, Optional[bytes]] = {}
    for path in paths:
        try:
            with open(path, "rb") as f:
                out[path] = f.read()
        except FileNotFoundError:
            out[path] = None
    return out


def _put_back(saved: Dict[str, Optional[bytes]]) -> None:
    for path, data in saved.items():
        if data is not None:
            with open(path, "wb") as f:
                f.write(data)
        elif os.path.exists(path):
            os.remove(path)
    for folder in sorted({os.path.dirname(p) for p in saved}, key=len, reverse=True):
        try:
            os.rmdir(folder)                                                    # a folder the attempt made, empty again
        except OSError:
            pass


def _champion_files(problem: str, name: str) -> List[str]:
    """what _record writes"""
    return [os.path.join(champions_dir(problem), "champions.json")] + [os.path.join(champions_dir(problem), name, f) for f in ("network.json", "architecture.svg")]


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
    rids = [f"{exp_id}.{arm}_s{s}" for s in card["seeds"]]
    _run_folders(problem, rids)
    tol = p["evaluate"].get("anchor_tolerance", 0.0) if tolerance is None else tolerance
    cfg_path = os.path.join(problem_dir(problem), p["config"])
    saved, done = _saved([cfg_path] + _champion_files(problem, f"{exp_id}.{arm}")), False
    try:
        write_json(cfg_path, spec.merged(read_json(cfg_path), card["arms"][arm].get("set", {})))  # the code is committed (start checks it): the arm's settings become the defaults
        _anchor(problem, rids, p, tol)
        facts = _facts(problem, exp_id, arm, card, decision, f"by hand: {by_hand}" if by_hand else "verdict KEEP")
        written = _record(problem, facts, "champion")
        paths_c = sorted({rel(cfg_path)} | {rel(x) for x in written})
        c = gitio.commit_paths(root, paths_c, f"adopt {exp_id}.{arm} ({problem}) as champion under {decision}\n\n{facts['how']}; settings of {tag} made the defaults, anchor test passed") if commit else ""
        done = True
    finally:
        if not done:                                                            # whatever stopped it before the commit: the working tree as it was
            _put_back(saved)
    return {"champion": facts["name"], "settings": card["arms"][arm].get("set", {}), "commit": c}


def register(problem: str, exp_id: str, arm: str, decision: str, role: str, commit: bool = True) -> Dict[str, Any]:
    """the facts of a measured network (an experiment of kind measure) as champion or comparison network, on the owner's decision"""
    if role not in ("champion", "comparison"):
        raise ResearchError("role: champion or comparison")
    if decision not in gates.decided(problem):
        raise ResearchError(f"{decision} is not a decided decision of {problem}")
    p = load_problem(problem)
    snap = gitio.tag_commit(repo(), f"{p['tag_prefix']}/{exp_id}")
    if not snap:
        raise ResearchError(f"{exp_id} has no snapshot {p['tag_prefix']}/{exp_id}: only a started measurement is registered")
    card = load_card(problem, exp_id, at=snap)
    if card.get("kind") != "measure":
        raise ResearchError(f"{exp_id} is not a measurement: a compared arm becomes champion through adopt")
    try:
        verdict = read_json(os.path.join(experiment_dir(problem, exp_id), "verdict.json"))
    except (OSError, ValueError):
        verdict = {}
    if verdict.get("verdict") != "DONE" or arm not in verdict.get("arms", {}):
        raise ResearchError(f"{exp_id} has no DONE verdict with the arm {arm}: `research judge {problem} {exp_id} --write` first")
    _run_folders(problem, [f"{exp_id}.{arm}_s{s}" for s in card["seeds"]])
    saved, done = _saved(_champion_files(problem, f"{exp_id}.{arm}")), False
    try:
        facts = _facts(problem, exp_id, arm, card, decision, f"registered as {role}")
        written = _record(problem, facts, role)
        c = gitio.commit_paths(repo(), [rel(x) for x in written], f"register {exp_id}.{arm} ({problem}) as {role} under {decision}") if commit else ""
        done = True
    finally:
        if not done:
            _put_back(saved)
    return {"name": facts["name"], "role": role, "commit": c}
