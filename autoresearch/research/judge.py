"""The verdict of an experiment, by the rule its snapshot holds.

For every arm other than the control: every run finished and evaluated? (else INCOMPLETE, nothing written) -> the control valid? (its score
reaches the floor of problem.json for every seed, and its seeds lie no further apart than the card's max_seed_spread; else INVALID) -> the
paired scene bootstrap of the pooled seeds (research/compute.py, in the snapshot with the problem's interpreter) -> the standard rule:

    gain = variant - control in the direction of the score (positive = better); e = the card's min_effect (default: the problem's band)
    TIE           |gain| < e: inside the band, whatever the interval  (also a KEEP whose guard broke: the gain and the guard are in the reason)
    KEEP          gain >= e and the interval of the gain lies above 0
    CUT           gain <= -e and the interval lies below 0
    INCONCLUSIVE  |gain| >= e but the interval includes 0: add a seed

A card may name a rule script instead (`rule.script`, in the experiment folder, so frozen in the snapshot): it gets the computed numbers and the
standard verdict as json on stdin and prints {"verdict", "reason"}."""
from __future__ import annotations

import json
import math
import os
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

from research import gitio, spec
from research.launch import load_card, load_problem
from research.paths import ResearchError, experiment_dir, interpreter, read_json, record_dir, rel, repo, runs_root, write_json
from research.worker import ensure_code

VERDICTS = ("KEEP", "CUT", "TIE", "INCONCLUSIVE", "INVALID")


def _arm_runs(problem: str, card: Dict[str, Any], arm: str) -> Dict[str, str]:
    """seed -> run id of an arm (an arm with 'from' points at another experiment's runs of the same seeds)"""
    ref = card["arms"][arm].get("from")
    exp, name = (ref.split(".") if ref else (card["id"], arm))
    return {str(s): f"{exp}.{name}_s{s}" for s in card["seeds"]}


def _results(problem: str, runs: Dict[str, str]) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    got, missing = {}, []
    for seed, rid in runs.items():
        try:
            run = read_json(os.path.join(record_dir(problem, rid), "run.json"))
            res = read_json(os.path.join(record_dir(problem, rid), "result.json"))
        except (OSError, ValueError, ResearchError):
            missing.append(f"{rid} (no result)")
            continue
        if run.get("status") != "ok":
            missing.append(f"{rid} ({run.get('status')})")
            continue
        got[seed] = {"run": rid, "result": res, "commit": run.get("commit", "")}
    return got, missing


def standard(gain: float, lo: float, hi: float, e: float) -> Tuple[str, str]:
    if not all(math.isfinite(x) for x in (gain, lo, hi)):
        return "INCONCLUSIVE", f"the gain or its interval is not defined ({gain}, [{lo}, {hi}])"
    if abs(gain) < e:
        return "TIE", f"gain {gain:+.4f} lies inside the band {e} (interval [{lo:+.4f}, {hi:+.4f}])"
    if gain > 0 and lo > 0:
        return "KEEP", f"gain {gain:+.4f} >= {e} and its interval [{lo:+.4f}, {hi:+.4f}] lies above 0"
    if gain < 0 and hi < 0:
        return "CUT", f"loss {gain:+.4f} beyond the band {e} and its interval [{lo:+.4f}, {hi:+.4f}] lies below 0"
    return "INCONCLUSIVE", f"gain {gain:+.4f} but its interval [{lo:+.4f}, {hi:+.4f}] includes 0: add a seed"


def _compute(problem: str, code: str, p: Dict[str, Any], query: Dict[str, Any]) -> Dict[str, Any]:
    env = dict(os.environ, RESEARCH_RUNS=runs_root(problem), RESEARCH_REPO=repo(),
               PYTHONPATH=os.pathsep.join([os.path.join(code, "problems", problem), os.path.join(code, "autoresearch"), code]))
    r = subprocess.run([interpreter(p["python"]), "-m", "research.compute"], cwd=os.path.join(code, "problems", problem), env=env,
                       input=json.dumps(query).encode(), capture_output=True)
    if r.returncode != 0:
        raise ResearchError(f"the computation of the numbers failed: {r.stderr.decode('utf-8', 'replace').strip()[-1500:]}")
    return json.loads(r.stdout.decode())


def judge(problem: str, exp_id: str, boot: int = 2000) -> Dict[str, Any]:
    root = repo()
    p_now = load_problem(problem)
    tag = f"{p_now['tag_prefix']}/{exp_id}"
    commit = gitio.tag_commit(root, tag)
    if not commit:
        raise ResearchError(f"{exp_id} has no snapshot {tag}: it was never started")
    p = load_problem(problem, at=commit)
    card = load_card(problem, exp_id, at=commit)
    if load_card(problem, exp_id) != card:
        raise ResearchError(f"the card of {exp_id} changed since its snapshot {tag}: the verdict follows the snapshot, so the records must too")
    score, better = p["score"]["metric"], p["score"]["better"]
    if card.get("kind") == "measure":
        return measure(problem, card, p, tag, commit)
    sign = 1.0 if better == "higher" else -1.0
    e = spec.min_effect(card, p)
    control = card["control"]
    c_runs, c_missing = _results(problem, _arm_runs(problem, card, control))
    out: Dict[str, Any] = {"experiment": exp_id, "problem": problem, "tag": tag, "snapshot": commit, "control": control, "score": score, "better": better,
                           "min_effect": e, "rule": card.get("rule", {}), "arms": {}, "by": "research judge", "date": time.strftime("%Y-%m-%d %H:%M")}
    incomplete = list(c_missing)
    arms = [a for a in card["arms"] if a != control]
    v_runs = {}
    for a in arms:
        v_runs[a], miss = _results(problem, _arm_runs(problem, card, a))
        incomplete += miss
    if incomplete:
        return {**out, "verdict": "INCOMPLETE", "missing": incomplete}
    # the control
    floor = p.get("control_floor", {}).get(score)
    c_scores = {s: r["result"]["score"]["value"] for s, r in c_runs.items()}
    problems = []
    if floor is not None:
        bad = {s: v for s, v in c_scores.items() if sign * (v - floor) < 0}
        if bad:
            problems.append(f"the control does not reach the floor {score} {floor} (seed: score {bad})")
    spread = card.get("rule", {}).get("max_seed_spread")
    if spread is not None and len(c_scores) > 1 and max(c_scores.values()) - min(c_scores.values()) > spread:
        problems.append(f"the control's seeds lie {max(c_scores.values()) - min(c_scores.values()):.4f} apart, more than {spread}")
    out["control_check"] = {"scores": c_scores, "floor": floor, "max_seed_spread": spread, "problems": problems}
    code = ensure_code(problem, commit)
    guards = p.get("guards", [])
    for a in arms:
        seeds = sorted(set(c_runs) & set(v_runs[a]), key=int)
        if problems:
            out["arms"][a] = {"verdict": "INVALID", "reason": "; ".join(problems)}
            continue
        q = {"score": score, "guards": [g["metric"] for g in guards], "boot": card.get("rule", {}).get("boot", boot),
             "control": {s: c_runs[s]["result"]["evaluation"]["id"] for s in seeds}, "variant": {s: v_runs[a][s]["result"]["evaluation"]["id"] for s in seeds}}
        num = _compute(problem, code, p, q)
        pooled = num["pooled"]
        gain = sign * pooled["diff"]
        lo, hi = (pooled["lo"], pooled["hi"]) if sign > 0 else (-pooled["hi"], -pooled["lo"])
        verdict, reason = standard(gain, lo, hi, e)
        broken = []
        for g in guards:
            gv = num["guards"][g["metric"]]
            s_g = 1.0 if gv["better"] == "high" else -1.0
            if not all(isinstance(gv[k], (int, float)) and math.isfinite(gv[k]) for k in ("control", "variant")):
                broken.append(f"{g['metric']} is not defined ({gv['control']} / {gv['variant']})")
                continue
            if "min" in g and gv["variant"] < g["min"]:
                broken.append(f"{g['metric']} {gv['variant']:.4f} < {g['min']}")
            if "max" in g and gv["variant"] > g["max"]:
                broken.append(f"{g['metric']} {gv['variant']:.4f} > {g['max']}")
            if "max_drop" in g and s_g * (gv["control"] - gv["variant"]) > g["max_drop"]:
                broken.append(f"{g['metric']} drops {abs(gv['variant'] - gv['control']):.4f} > {g['max_drop']}")
        if verdict == "KEEP" and broken:
            verdict, reason = "TIE", f"not adopted: {reason}, but a guard broke: {'; '.join(broken)}"
        arm_out = {"verdict": verdict, "reason": reason, "gain": gain, "interval": [lo, hi], "seeds": seeds, "numbers": num,
                   "prediction": card["prediction"], "prediction_hit": abs(gain - card["prediction"]) <= e,
                   "guards_broken": broken}
        script = card.get("rule", {}).get("script")
        if script:
            arm_out["standard"] = {"verdict": verdict, "reason": reason}
            res = subprocess.run([interpreter(p["python"]), os.path.join(code, rel(experiment_dir(problem, exp_id)), script)],
                                 input=json.dumps(arm_out, default=float).encode(), capture_output=True, cwd=os.path.join(code, "problems", problem))
            try:
                v = json.loads(res.stdout.decode())
                if v.get("verdict") not in VERDICTS:
                    raise ValueError(f"verdict {v.get('verdict')!r}")
                arm_out.update(verdict=v["verdict"], reason=f"rule script {script}: {v.get('reason', '')}")
            except (ValueError, AttributeError) as err:
                raise ResearchError(f"the rule script {script} gave no verdict ({err}): {res.stderr.decode('utf-8', 'replace')[-800:]}") from None
        out["arms"][a] = arm_out
    return {**out, "verdict": "DONE"}


def measure(problem: str, card: Dict[str, Any], p: Dict[str, Any], tag: str, commit: str) -> Dict[str, Any]:
    """a measurement: the numbers of every arm and seed, no comparison and no verdict"""
    out: Dict[str, Any] = {"experiment": card["id"], "problem": problem, "tag": tag, "snapshot": commit, "kind": "measure", "score": p["score"]["metric"],
                           "arms": {}, "by": "research judge", "date": time.strftime("%Y-%m-%d %H:%M")}
    missing: List[str] = []
    for a in card["arms"]:
        got, miss = _results(problem, _arm_runs(problem, card, a))
        missing += miss
        out["arms"][a] = {"verdict": "MEASURED", "seeds": {s: {"run": r["run"], "numbers": r["result"]["numbers"], "frames": r["result"]["frames"],
                                                               "cost": r["result"].get("cost")} for s, r in got.items()}}
    return {**out, "verdict": "INCOMPLETE", "missing": missing} if missing else {**out, "verdict": "DONE"}


def write(problem: str, exp_id: str, verdict: Dict[str, Any], commit: bool = True) -> str:
    """verdict.json, the ledger, and one commit of the experiment's records; -> the commit ('' when not committed)"""
    from research import index
    folder = experiment_dir(problem, exp_id)
    write_json(os.path.join(folder, "verdict.json"), verdict)
    ledger = index.write(problem)
    if not commit:
        return ""
    summary = ", ".join(f"{a}: {v['verdict']}" for a, v in verdict["arms"].items())
    return gitio.commit_paths(repo(), [rel(folder), rel(ledger)], f"{exp_id} ({problem}): verdict {summary}\n\nresearch judge, rule of {verdict['tag']}")
