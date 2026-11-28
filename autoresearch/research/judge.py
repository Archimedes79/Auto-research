"""The verdict of an experiment, by the rule its snapshot holds.

For every arm other than the control: every run finished and evaluated? (else INCOMPLETE, nothing written; a run a kill rule stopped makes its
arm CUT, the control INVALID) -> the control valid? (its score reaches the floor of problem.json for every seed, and its seeds lie no further
apart than the card's max_seed_spread; else INVALID) -> scored alike? (split, frames, ground truth, evaluator and metrics the same in both
arms; else INVALID) -> the paired scene bootstrap of the pooled seeds (research/compute.py, in the snapshot with the problem's interpreter)
-> the standard rule (a KEEP of one seed only, or whose seeds disagree in sign, is INCONCLUSIVE: `seed_check`):

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
from research.paths import (CORE, ResearchError, child_env, core_step, experiment_dir, interpreter, module_path, read_json, record_dir, rel, repo, runs_root,
                            write_json)
from research.worker import ensure_code

VERDICTS = ("KEEP", "CUT", "TIE", "INCONCLUSIVE", "INVALID")
JUDGE_FILES = ("research/judge.py", "research/spec.py", "research/launch.py", "research/paths.py", "research/gitio.py", "research/worker.py")
SNAPSHOT_FILES = ("research/compute.py", "Evaluation/stats.py")


def judge_code() -> Dict[str, Any]:
    """the code of a verdict, recorded in it: the files that run from the working tree while judging (the rule, the card and problem readers,
    the snapshot's unpacking) with one hash (12 hex); compute.py and the bootstrap run from the snapshot (and so does a rule script), whose
    commit the verdict records already"""
    import hashlib
    h = hashlib.sha256()
    for f in JUDGE_FILES:
        with open(os.path.join(CORE, f), "rb") as fh:
            h.update(f.encode() + b"\0" + fh.read())
    return {"hash": h.hexdigest()[:12], "files": [f"autoresearch/{f}" for f in JUDGE_FILES], "from_snapshot": [f"autoresearch/{f}" for f in SNAPSHOT_FILES]}


def _arm_runs(problem: str, card: Dict[str, Any], arm: str) -> Dict[str, str]:
    """seed -> run id of an arm (an arm with 'from' points at another experiment's runs of the same seeds)"""
    ref = card["arms"][arm].get("from")
    exp, name = (ref.split(".") if ref else (card["id"], arm))
    return {str(s): f"{exp}.{name}_s{s}" for s in card["seeds"]}


def _results(problem: str, runs: Dict[str, str]) -> Tuple[Dict[str, Dict[str, Any]], List[str], List[str]]:
    """(seed -> the finished run's result, runs that are not finished, runs a kill rule stopped)"""
    got, missing, killed = {}, [], []
    for seed, rid in runs.items():
        try:
            run = read_json(os.path.join(record_dir(problem, rid), "run.json"))
        except (OSError, ValueError, ResearchError):
            missing.append(f"{rid} (not started)")
            continue
        if run.get("status") == "killed":
            why = next((s.get("why", "") for s in run.get("steps", {}).values() if s.get("status") == "killed"), "")
            killed.append(f"{rid} ({why})")
            continue
        try:
            res = read_json(os.path.join(record_dir(problem, rid), "result.json"))
        except (OSError, ValueError):
            res = None
        if run.get("status") != "ok" or res is None:
            missing.append(f"{rid} ({run.get('status')}{', no result' if res is None else ''})")
            continue
        got[seed] = {"run": rid, "result": res, "commit": run.get("commit", "")}
    return got, missing, killed


COMPARABLE = (("split", lambda r: r["split"]), ("frames", lambda r: r["frames"]), ("ground truth", lambda r: r["evaluation"].get("gt")),
              ("evaluator", lambda r: r["evaluation"].get("evaluator")), ("metrics", lambda r: r["evaluation"].get("metrics")))


def incomparable(runs: List[Dict[str, Any]]) -> List[str]:
    """what differs between the results of the runs of one comparison (they must be scored on the same frames, ground truth and evaluator).
    "ground truth" is the name a source gives it; its content signature is compared between the records that hold one (`unsigned`)"""
    out = []
    for name, get in COMPARABLE:
        vals = {json.dumps(get(r["result"]), sort_keys=True, default=str) for r in runs}
        if len(vals) > 1:
            out.append(f"{name} differs: {sorted(vals)[:3]}")
    signatures = {r["result"]["evaluation"].get("gt_signature") for r in runs} - {None, ""}
    if len(signatures) > 1:
        out.append(f"ground truth signature differs: {sorted(signatures)[:3]}")
    return out


def unsigned(runs: List[Dict[str, Any]]) -> List[str]:
    """the runs whose record holds no ground-truth signature (records of before it was kept): their ground truth is compared by name only"""
    return [r.get("run", "?") for r in runs if not r["result"]["evaluation"].get("gt_signature")]


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


def seed_check(verdict: str, reason: str, per_seed: Dict[str, float]) -> Tuple[str, str]:
    """a KEEP needs two seeds that agree: the interval is a bootstrap over the scenes, so only a second seed shows what training alone moves.
    A KEEP of one seed, or of seeds that disagree in sign, is INCONCLUSIVE; every other verdict stays"""
    if verdict != "KEEP":
        return verdict, reason
    if len(per_seed) < 2:
        return "INCONCLUSIVE", f"{reason}, but from one seed only: add a seed"
    if any(g <= 0 for g in per_seed.values()):
        return "INCONCLUSIVE", f"{reason}, but the seeds disagree (gain per seed {per_seed}): add a seed"
    return verdict, reason


def gain_interval(pooled: Dict[str, float], sign: float) -> Tuple[float, float, float]:
    """(gain, lo, hi) in the direction of the score: positive = better, also for a score where lower is better"""
    gain = sign * pooled["diff"]
    lo, hi = (pooled["lo"], pooled["hi"]) if sign > 0 else (-pooled["hi"], -pooled["lo"])
    return gain, lo, hi


def guards_broken(guards: List[Dict[str, Any]], numbers: Dict[str, Dict[str, Any]]) -> List[str]:
    """the guards of problem.json the variant breaks (numbers: metric -> control, variant, better); an undefined number breaks its guard"""
    broken = []
    for g in guards:
        gv = numbers[g["metric"]]
        s_g = 1.0 if gv["better"] == "high" else -1.0
        if not all(isinstance(gv[k], (int, float)) and math.isfinite(gv[k]) for k in ("control", "variant")):
            broken.append(f"{g['metric']} is not defined ({gv['control']} / {gv['variant']})")
            continue
        if "min" in g and gv["variant"] < g["min"]:
            broken.append(f"{g['metric']} {gv['variant']:.4f} < {g['min']}")
        if "max" in g and gv["variant"] > g["max"]:
            broken.append(f"{g['metric']} {gv['variant']:.4f} > {g['max']}")
        if "max_drop" in g and s_g * (gv["control"] - gv["variant"]) > g["max_drop"]:
            broken.append(f"{g['metric']} worse by {abs(gv['variant'] - gv['control']):.4f} > {g['max_drop']}")
    return broken


def control_problems(scores: Dict[str, Any], floor: Optional[float], spread: Optional[float], sign: float, score: str) -> List[str]:
    """why the control is not valid: a seed below the floor of problem.json, or seeds further apart than the card's max_seed_spread; an
    undefined score fails both"""
    ok = {s: v for s, v in scores.items() if isinstance(v, (int, float)) and math.isfinite(v)}
    out = []
    if floor is not None:
        bad = {s: v for s, v in scores.items() if s not in ok or sign * (v - floor) < 0}
        if bad:
            out.append(f"the control does not reach the floor {score} {floor} (seed: score {bad})")
    if spread is not None and len(scores) > 1:
        if len(ok) < len(scores):
            out.append(f"the control's seeds are not all defined ({scores}), so their spread is not known")
        elif max(ok.values()) - min(ok.values()) > spread:
            out.append(f"the control's seeds lie {max(ok.values()) - min(ok.values()):.4f} apart, more than {spread}")
    return out


def _snapshot_env(problem: str, code: str) -> Dict[str, str]:
    """the environment of the computation and the rule script: the snapshot's code, nothing of the caller's trees"""
    return child_env(RESEARCH_RUNS=runs_root(problem), RESEARCH_REPO=repo(), PYTHONPATH=module_path(code, problem))


def _compute(problem: str, code: str, p: Dict[str, Any], query: Dict[str, Any]) -> Dict[str, Any]:
    r = subprocess.run(core_step(p["python"], "research.compute"), cwd=os.path.join(code, "problems", problem), env=_snapshot_env(problem, code),
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
    frozen, now = load_card(problem, exp_id, at=commit), load_card(problem, exp_id)
    if not spec.same_rule(frozen, now):
        raise ResearchError(f"the card of {exp_id} changed since its snapshot {tag}: the verdict follows the snapshot, so the records must too (only seeds may be added)")
    card = dict(frozen, seeds=now["seeds"])
    score, better = p["score"]["metric"], p["score"]["better"]
    if card.get("kind") == "measure":
        return measure(problem, card, p, tag, commit)
    sign = 1.0 if better == "higher" else -1.0
    e = spec.min_effect(card, p)
    control = card["control"]
    c_runs, c_missing, c_killed = _results(problem, _arm_runs(problem, card, control))
    out: Dict[str, Any] = {"experiment": exp_id, "problem": problem, "tag": tag, "snapshot": commit, "control": control, "score": score, "better": better,
                           "min_effect": e, "rule": card.get("rule", {}), "boot": card.get("rule", {}).get("boot", boot), "seeds": card["seeds"],
                           "arms": {}, "by": "research judge", "judge_code": judge_code(), "date": time.strftime("%Y-%m-%d %H:%M")}
    incomplete = list(c_missing)
    arms = [a for a in card["arms"] if a != control]
    v_runs, v_killed = {}, {}
    for a in arms:
        v_runs[a], miss, v_killed[a] = _results(problem, _arm_runs(problem, card, a))
        incomplete += miss
    if incomplete:
        return {**out, "verdict": "INCOMPLETE", "missing": incomplete}
    # the control
    floor = p.get("control_floor", {}).get(score)
    c_scores = {s: r["result"]["score"]["value"] for s, r in c_runs.items()}
    spread = card.get("rule", {}).get("max_seed_spread")
    problems = ([f"a kill rule stopped the control: {', '.join(c_killed)}"] if c_killed else []) + control_problems(c_scores, floor, spread, sign, score)
    foreign = [r["run"] for runs_ in [c_runs] + list(v_runs.values()) for r in runs_.values()
               if r["run"].startswith(exp_id + ".") and (r["commit"] != commit or r["result"].get("identity", {}).get("commit") != commit)]
    if foreign:
        problems.append(f"runs that did not execute the snapshot {commit[:10]}: {foreign}")
    out["control_check"] = {"scores": c_scores, "floor": floor, "max_seed_spread": spread, "problems": problems}
    code = ensure_code(problem, commit)
    guards = p.get("guards", [])
    for a in arms:
        seeds = sorted(set(c_runs) & set(v_runs[a]), key=int)
        if problems:
            out["arms"][a] = {"verdict": "INVALID", "reason": "; ".join(problems)}
            continue
        if v_killed[a]:
            out["arms"][a] = {"verdict": "CUT", "reason": f"a kill rule of the card stopped the arm: {', '.join(v_killed[a])}"}
            continue
        differs = incomparable([c_runs[s] for s in seeds] + [v_runs[a][s] for s in seeds])
        if differs:
            out["arms"][a] = {"verdict": "INVALID", "reason": "the arms are not scored alike: " + "; ".join(differs)}
            continue
        q = {"score": score, "guards": [g["metric"] for g in guards], "boot": card.get("rule", {}).get("boot", boot),
             "control": {s: c_runs[s]["result"]["evaluation"]["id"] for s in seeds}, "variant": {s: v_runs[a][s]["result"]["evaluation"]["id"] for s in seeds}}
        num = _compute(problem, code, p, q)
        gain, lo, hi = gain_interval(num["pooled"], sign)
        verdict, reason = standard(gain, lo, hi, e)
        per_seed = {s: sign * v["diff"] for s, v in num["seeds"].items()}
        kept = verdict == "KEEP"
        verdict, reason = seed_check(verdict, reason, per_seed)
        broken = guards_broken(guards, num["guards"])
        if verdict == "KEEP" and broken:
            verdict, reason = "TIE", f"not adopted: {reason}, but a guard broke: {'; '.join(broken)}"
        elif kept and broken:                                    # the seeds stopped it first: the reason still names the guard
            reason = f"{reason}; a guard broke too: {'; '.join(broken)}"
        arm_out = {"verdict": verdict, "reason": reason, "gain": gain, "interval": [lo, hi], "seeds": seeds, "numbers": num,
                   "prediction": card["prediction"], "prediction_hit": abs(gain - card["prediction"]) <= e,
                   "guards_broken": broken}
        old = unsigned([c_runs[s] for s in seeds] + [v_runs[a][s] for s in seeds])
        if old:
            arm_out["warnings"] = [f"ground truth compared by name only: the records of {', '.join(old)} hold no signature"]
        script = card.get("rule", {}).get("script")
        if script:
            arm_out["standard"] = {"verdict": verdict, "reason": reason}
            res = subprocess.run([interpreter(p["python"]), os.path.join(code, rel(experiment_dir(problem, exp_id)), script)],
                                 input=json.dumps(arm_out, default=float).encode(), capture_output=True, cwd=os.path.join(code, "problems", problem),
                                 env=_snapshot_env(problem, code))
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
    """a measurement: the score and guard numbers of every arm and seed (every number: the run's result.json, named), no comparison"""
    out: Dict[str, Any] = {"experiment": card["id"], "problem": problem, "tag": tag, "snapshot": commit, "kind": "measure", "score": p["score"]["metric"],
                           "arms": {}, "by": "research judge", "judge_code": judge_code(), "date": time.strftime("%Y-%m-%d %H:%M")}
    missing: List[str] = []
    for a in card["arms"]:
        got, miss, killed = _results(problem, _arm_runs(problem, card, a))
        miss += killed
        missing += miss
        out["arms"][a] = {"verdict": "MEASURED", "seeds": {s: {"run": r["run"], "numbers": spec.key_numbers(p, r["result"]["numbers"]),
                                                               "frames": r["result"]["frames"], "cost": r["result"].get("cost"),
                                                               "result": rel(os.path.join(record_dir(problem, r["run"]), "result.json"))}
                                                           for s, r in got.items()}}
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
