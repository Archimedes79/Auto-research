"""new, check, start, test: from a card to running workers.

start = check -> the snapshot (one commit + tag <prefix>/E<n> per experiment; a later start of the same experiment reuses it and refuses a card
that changed since) -> per run config.json and run.json in the records -> one worker process per run. Runs that exist are not started again
(a stopped or failed run is continued with `resume`)."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from research import gates, gitio, spec
from research.paths import (CORE, ResearchError, exp_number, experiment_dir, problem_dir, read_json, record_dir, rel, repo, run_dir, run_id,
                            runs_root, write_json)

SNAPSHOT_PATHS = ("autoresearch", "GtGenerator", "datasets")       # besides the problem's own folder: the code a run may import


def load_problem(problem: str, root: Optional[str] = None, at: Optional[str] = None) -> Dict[str, Any]:
    """problem.json of the working tree, or of commit `at`; checked"""
    path = os.path.join(problem_dir(problem, root), "problem.json")
    try:
        p = json.loads(gitio.show(root or repo(), at, rel(path, root)) if at else open(path, encoding="utf-8").read())
    except ValueError as e:
        raise ResearchError(f"{path}: not json ({e})") from None
    errs = spec.problem_errors(p, problem)
    if errs:
        raise ResearchError("\n".join(errs))
    return p


def load_card(problem: str, exp_id: str, root: Optional[str] = None, at: Optional[str] = None) -> Dict[str, Any]:
    path = os.path.join(experiment_dir(problem, exp_id, root), "experiment.md")
    text = gitio.show(root or repo(), at, rel(path, root)) if at else open(path, encoding="utf-8").read()
    card = spec.parse_card(text, rel(path, root))
    if card.get("id") != exp_id:
        raise ResearchError(f"{rel(path, root)}: the card says id {card.get('id')!r}, the folder {exp_id}")
    return card


def base_config(problem: str, p: Dict[str, Any], root: Optional[str] = None, at: Optional[str] = None) -> Dict[str, Any]:
    path = os.path.join(problem_dir(problem, root), p["config"])
    try:
        cfg = json.loads(gitio.show(root or repo(), at, rel(path, root)) if at else open(path, encoding="utf-8").read())
    except (OSError, ResearchError) as e:
        raise ResearchError(f"the base config {p['config']}: {e}") from None
    missing = [k for k in spec.CONFIG_KEYS if k not in cfg]
    if missing:
        raise ResearchError(f"the base config {p['config']} lacks {missing}")
    return cfg


# ------------------------------------------------------------------------------------------------ new
TEMPLATE = os.path.join(CORE, "templates", "experiment.md")


def new(problem: str, title: str, slug: str) -> str:
    """the next experiment folder with the card template; -> its path"""
    if not re.match(r"^[a-z0-9][a-z0-9_]*$", slug):
        raise ResearchError(f"slug {slug!r}: lower-case letters, digits and _")
    p = load_problem(problem)
    base = os.path.join(problem_dir(problem), "experiments")
    os.makedirs(base, exist_ok=True)
    taken = [exp_number(m.group(1)) for d in os.listdir(base) for m in [re.match(r"^(E\d+)(_|$)", d)] if m]
    n = max(taken + [p.get("first_experiment", 1) - 1]) + 1
    with open(TEMPLATE, encoding="utf-8") as f:
        text = f.read()
    while True:                                                  # mkdir is atomic: two agents never get the same id
        path = os.path.join(base, f"E{n:03d}_{slug}")
        try:
            os.mkdir(path)
            break
        except FileExistsError:
            n += 1
    with open(os.path.join(path, "experiment.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text.replace("E000", f"E{n}").replace("<title>", title))
    return path


# ------------------------------------------------------------------------------------------------ check
def runs_of(card: Dict[str, Any], arms: Optional[Sequence[str]] = None, seeds: Optional[Sequence[int]] = None) -> List[Tuple[str, str, int]]:
    """(run id, arm, seed) of the runs this experiment makes (arms with 'from' reuse stored runs and make none)"""
    arms = list(arms or card["arms"])
    seeds = list(seeds if seeds is not None else card["seeds"])
    bad = [a for a in arms if a not in card["arms"]] + [str(s) for s in seeds if s not in card["seeds"]]
    if bad:
        raise ResearchError(f"not in the card: {', '.join(bad)}")
    return [(run_id(card["id"], a, s), a, s) for s in seeds for a in arms if "from" not in card["arms"][a]]


def check(problem: str, exp_id: str) -> Dict[str, Any]:
    """everything start would refuse, without starting anything: -> {errors, runs, tag}"""
    errors: List[str] = []
    p = load_problem(problem)
    card = load_card(problem, exp_id)
    cfg = base_config(problem, p)
    errors += spec.card_errors(card, p, cfg)
    errors += [x for x in [gates.hold_refusal(problem)] if x]
    for name, arm in card["arms"].items():
        for ref in [arm.get("from")] + list(arm.get("needs", {}).values()):
            if ref:
                other = ref.split(".")[0]
                try:
                    experiment_dir(problem, other)
                except ResearchError as e:
                    errors.append(f"arm {name}: {e}")
    if card.get("rule", {}).get("script"):
        script = os.path.join(experiment_dir(problem, exp_id), card["rule"]["script"])
        if not os.path.isfile(script):
            errors.append(f"the rule script {rel(script)} does not exist")
    tag = f"{p['tag_prefix']}/{exp_id}"
    commit = gitio.tag_commit(repo(), tag)
    if commit:
        try:
            frozen = load_card(problem, exp_id, at=commit)
            if frozen != card:
                errors.append(f"the card changed since its snapshot {tag}: the rule is what the tag holds; a changed experiment is a new experiment")
        except ResearchError as e:
            errors.append(f"the snapshot {tag}: {e}")
    return {"errors": errors, "runs": [r for r, _, _ in runs_of(card)] if not errors else [], "tag": tag, "snapshot": commit[:12]}


# ------------------------------------------------------------------------------------------------ start
def start(problem: str, exp_id: str, arms: Optional[Sequence[str]] = None, seeds: Optional[Sequence[int]] = None, spawn: bool = True) -> Dict[str, Any]:
    c = check(problem, exp_id)
    if c["errors"]:
        raise ResearchError("refused:\n  " + "\n  ".join(c["errors"]))
    root = repo()
    p = load_problem(problem)
    card = load_card(problem, exp_id)
    todo = runs_of(card, arms, seeds)
    commit = gitio.tag_commit(root, c["tag"])
    if not commit:
        paths = [x for x in SNAPSHOT_PATHS if os.path.isdir(os.path.join(root, x))] + [rel(problem_dir(problem))]
        commit = gitio.snapshot(root, paths, f"{exp_id} ({problem}): {card['title']}\n\nsnapshot of research start; the runs execute this commit", c["tag"])
    p = load_problem(problem, at=commit)                         # from here on everything is read from the snapshot
    card = load_card(problem, exp_id, at=commit)
    cfg = base_config(problem, p, at=commit)
    started, skipped = [], []
    for rid, arm, seed in todo:
        rec = record_dir(problem, rid)
        if os.path.exists(os.path.join(rec, "run.json")):
            skipped.append(f"{rid} ({read_json(os.path.join(rec, 'run.json')).get('status')})")
            continue
        write_json(os.path.join(rec, "config.json"), spec.compose(cfg, card, arm, seed, rid))
        write_json(os.path.join(rec, "run.json"), {"run": rid, "problem": problem, "experiment": exp_id, "arm": arm, "seed": seed,
                                                    "commit": commit, "tag": c["tag"], "steps": {s: {"status": "pending"} for s in spec.steps_of(card, p)},
                                                    "status": "queued", "created": time.strftime("%Y-%m-%d %H:%M:%S")})
        started.append(rid)
        if spawn:
            spawn_worker(problem, rid)
    return {"tag": c["tag"], "commit": commit[:12], "started": started, "skipped": skipped}


def spawn_worker(problem: str, rid: str, extra: Sequence[str] = ()) -> None:
    """one detached worker process for one run (this interpreter: the worker needs only the standard library)"""
    out = run_dir(problem, rid)
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, RESEARCH_REPO=repo(), PYTHONPATH=CORE)
    with open(os.path.join(out, "worker.log"), "ab") as log:
        subprocess.Popen([sys.executable, "-m", "research", "worker", problem, rid, *extra], cwd=CORE, env=env, stdout=log, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)


# ------------------------------------------------------------------------------------------------ TEST
def test(problem: str, rid: str, spawn: bool = True) -> Dict[str, Any]:
    """evaluate a finished run once on the holdout split: only under a row of the TEST log with a decided decision"""
    why = gates.test_refusal(problem, rid)
    if why:
        raise ResearchError(f"TEST refused: {why}")
    rec = record_dir(problem, rid)
    run = read_json(os.path.join(rec, "run.json"))
    if run.get("status") != "ok":
        raise ResearchError(f"{rid} is {run.get('status')}: only a finished run is evaluated on TEST")
    if os.path.exists(os.path.join(rec, "test.json")):
        raise ResearchError(f"{rid} has a TEST result already: a TEST use is once")
    if spawn:
        spawn_worker(problem, rid, ["--test"])
    return {"run": rid, "started": spawn}


def runs_root_of(problem: str) -> str:
    return runs_root(problem)
