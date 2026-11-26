"""The two things an author writes: problem.json (what the problem is and how a result counts) and the card of an experiment (the json block in
experiment.md). Both are checked here, strictly: an unknown key is a typo until it is in the schema.

A card's arms change settings of the base config only; a setting the base config does not have is refused, so a new option enters the code
default-off in the base config first (and the experiment's snapshot holds both)."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from research.paths import NAME, ResearchError, exp_number

# key -> (type, required)
PROBLEM_KEYS = {
    "name": (str, True), "tag_prefix": (str, True), "python": (str, True), "config": (str, True),
    "steps": (dict, True), "default_steps": (list, True), "score": (dict, True), "evaluate": (dict, True),
    "guards": (list, False), "control_floor": (dict, False), "holdout": (list, False), "protected": (list, False),
    "data": (str, False), "first_experiment": (int, False), "test_steps": (list, False), "note": (str, False),
}
# test_steps: the steps a TEST evaluation of a finished run runs (default ["evaluate"]; a problem that predicts in a step of its own names it too)
CONFIG_KEYS = ("name", "seed", "out_root")                 # every base config has them: the run's name, its seed and where it writes
STEP_KEYS = {"run": ((list, str), True), "gpus": (int, False), "note": (str, False)}
SCORE_KEYS = {"metric": (str, True), "better": (str, True), "band": ((int, float), True)}
EVALUATE_KEYS = {"split": (str, True), "metrics": (list, True), "device": (str, False), "frames": (str, False), "anchor_tolerance": ((int, float), False)}
# frames: which frames of the split are scored (an Evaluation frame spec such as "spread:1000:val"; default: the whole split);
# anchor_tolerance: how far an evaluation repeated with the same weights may differ (0 = bit for bit; a GPU that is not deterministic needs more)
GUARD_KEYS = {"metric": (str, True), "better": (str, False), "min": ((int, float), False), "max": ((int, float), False),
              "max_drop": ((int, float), False)}

CARD_KEYS = {
    "id": (str, True), "title": (str, True), "question": (str, True), "hypothesis": (str, True), "prediction": ((int, float), False),
    "kind": (str, False), "control": (str, False), "arms": (dict, True), "seeds": (list, True), "rule": (dict, False), "steps": (list, False),
    "kill": (list, False), "split": (str, False), "note": (str, False),
}
KINDS = ("compare", "measure")                              # compare: arms against the control, a verdict; measure: the numbers of each arm, no verdict
ARM_KEYS = {"set": (dict, False), "from": (str, False), "needs": (dict, False), "note": (str, False)}
RULE_KEYS = {"min_effect": ((int, float), False), "max_seed_spread": ((int, float), False), "script": (str, False), "boot": (int, False)}
KILL_KEYS = {"metric": (str, True), "after_step": (int, True), "above": ((int, float), False), "below": ((int, float), False)}
CORE_STEPS = {"evaluate"}                                   # a step whose "run" is one of these is the core's code, not the problem's
FROM = re.compile(r"^(E\d+)\.([a-z][a-z0-9]*)$")            # an arm that reuses the runs of another experiment's arm


def _check(obj: Any, keys: Dict[str, Tuple[Any, bool]], where: str) -> List[str]:
    if not isinstance(obj, dict):
        return [f"{where}: must be an object"]
    errs = [f"{where}: unknown key '{k}' (known: {', '.join(sorted(keys))})" for k in obj if k not in keys]
    for k, (typ, required) in keys.items():
        if k not in obj:
            if required:
                errs.append(f"{where}: '{k}' is missing")
        elif not isinstance(obj[k], typ) or (isinstance(obj[k], bool) and typ is not bool):
            errs.append(f"{where}: '{k}' has the wrong type ({type(obj[k]).__name__})")
    return errs


def problem_errors(p: Dict[str, Any], folder_name: str = "") -> List[str]:
    errs = _check(p, PROBLEM_KEYS, "problem.json")
    if errs:
        return errs
    if folder_name and p["name"] != folder_name:
        errs.append(f"problem.json: name {p['name']!r} is not the folder name {folder_name!r}")
    if not NAME.match(p["name"]) or not NAME.match(p["tag_prefix"]):
        errs.append("problem.json: name and tag_prefix are lower-case letters, digits and _")
    for name, step in p["steps"].items():
        errs += _check(step, STEP_KEYS, f"problem.json steps.{name}")
        if isinstance(step, dict) and isinstance(step.get("run"), str) and step["run"] not in CORE_STEPS:
            errs.append(f"problem.json steps.{name}: 'run' is a list of words, or one of {sorted(CORE_STEPS)}")
    for name in p["default_steps"]:
        if name not in p["steps"]:
            errs.append(f"problem.json default_steps: no step {name!r}")
    errs += _check(p["score"], SCORE_KEYS, "problem.json score")
    if p["score"].get("better") not in ("higher", "lower"):
        errs.append("problem.json score.better: 'higher' or 'lower'")
    errs += _check(p["evaluate"], EVALUATE_KEYS, "problem.json evaluate")
    for i, g in enumerate(p.get("guards", [])):
        errs += _check(g, GUARD_KEYS, f"problem.json guards[{i}]")
        if isinstance(g, dict) and sum(k in g for k in ("min", "max", "max_drop")) != 1:
            errs.append(f"problem.json guards[{i}]: exactly one of min, max, max_drop")
    if p["evaluate"].get("split") in p.get("holdout", []):
        errs.append("problem.json: the evaluation split is a holdout split")
    return errs


def card_errors(card: Dict[str, Any], problem: Dict[str, Any], base_config: Optional[Dict[str, Any]] = None) -> List[str]:
    errs = _check(card, CARD_KEYS, "card")
    if errs:
        return errs
    try:
        exp_number(card["id"])
    except ResearchError as e:
        errs.append(f"card: {e}")
    kind = card.get("kind", "compare")
    if kind not in KINDS:
        errs.append(f"card: kind {kind!r} is one of {KINDS}")
    elif kind == "compare":
        if card.get("control") not in card["arms"]:
            errs.append(f"card: the control {card.get('control')!r} is not an arm")
        if len(card["arms"]) < 2:
            errs.append("card: a comparison needs the control and at least one other arm")
        if "prediction" not in card:
            errs.append("card: a comparison needs a prediction (the expected gain of the score over the control)")
    elif not card["arms"]:
        errs.append("card: a measurement needs at least one arm")
    if not card["seeds"] or not all(isinstance(s, int) and not isinstance(s, bool) and s >= 0 for s in card["seeds"]) or len(set(card["seeds"])) != len(card["seeds"]):
        errs.append("card: seeds are distinct integers >= 0")
    for name, arm in card["arms"].items():
        if not re.match(r"^[a-z][a-z0-9]*$", name):
            errs.append(f"card: arm name {name!r}: lower-case letters and digits, starting with a letter")
        errs += _check(arm, ARM_KEYS, f"card arms.{name}")
        if not isinstance(arm, dict):
            continue
        if "from" in arm:
            if not FROM.match(arm["from"]):
                errs.append(f"card arms.{name}.from: E<n>.<arm>, got {arm['from']!r}")
            if set(arm) - {"from", "note"}:
                errs.append(f"card arms.{name}: an arm with 'from' reuses stored runs and sets nothing")
        if base_config is not None:
            for key in unknown_settings(arm.get("set", {}), base_config):
                errs.append(f"card arms.{name}.set: {key!r} is not a setting of the base config (a new option goes into the base config, default off)")
            for key in arm.get("needs", {}):
                if key not in base_config:
                    errs.append(f"card arms.{name}.needs: {key!r} is not a setting of the base config")
        for key, ref in arm.get("needs", {}).items():
            if not isinstance(ref, str) or not FROM.match(ref):
                errs.append(f"card arms.{name}.needs.{key}: E<n>.<arm> (the run of the same seed), got {ref!r}")
    rule = card.get("rule", {})
    errs += _check(rule, RULE_KEYS, "card rule")
    for step in card.get("steps", problem.get("default_steps", [])):
        if step not in problem.get("steps", {}):
            errs.append(f"card steps: no step {step!r} in problem.json")
    for i, k in enumerate(card.get("kill", [])):
        errs += _check(k, KILL_KEYS, f"card kill[{i}]")
        if isinstance(k, dict) and ("above" in k) == ("below" in k):
            errs.append(f"card kill[{i}]: exactly one of above, below")
    split = card.get("split", problem.get("evaluate", {}).get("split"))
    if split in problem.get("holdout", []):
        errs.append(f"card: the split {split!r} is a holdout split; the verdict of an experiment is made on {problem['evaluate']['split']!r}")
    return errs


def parse_card(text: str, where: str = "experiment.md") -> Dict[str, Any]:
    """the one fenced ```json card block of an experiment.md"""
    blocks = re.findall(r"^```json card\s*\n(.*?)\n```\s*$", text, flags=re.S | re.M)
    if len(blocks) != 1:
        raise ResearchError(f"{where}: needs exactly one ```json card block, found {len(blocks)}")
    try:
        card = json.loads(blocks[0])
    except ValueError as e:
        raise ResearchError(f"{where}: the card block is not json ({e})") from None
    if not isinstance(card, dict):
        raise ResearchError(f"{where}: the card block must be an object")
    return card


def steps_of(card: Dict[str, Any], problem: Dict[str, Any]) -> List[str]:
    return list(card.get("steps") or problem["default_steps"])


def split_of(card: Dict[str, Any], problem: Dict[str, Any]) -> str:
    return card.get("split") or problem["evaluate"]["split"]


def frames_of(card: Dict[str, Any], problem: Dict[str, Any]) -> str:
    """the frames an evaluation scores: the problem's frame spec, or the whole split a card names"""
    return card["split"] if card.get("split") else problem["evaluate"].get("frames") or problem["evaluate"]["split"]


def min_effect(card: Dict[str, Any], problem: Dict[str, Any]) -> float:
    return float(card.get("rule", {}).get("min_effect", problem["score"]["band"]))


def unknown_settings(settings: Dict[str, Any], base: Dict[str, Any], prefix: str = "") -> List[str]:
    """the settings (nested: 'model.width') that the base config does not have"""
    out = []
    for k, v in settings.items():
        if k not in base:
            out.append(prefix + k)
        elif isinstance(v, dict) and isinstance(base[k], dict):
            out += unknown_settings(v, base[k], f"{prefix}{k}.")
    return out


def merged(base: Dict[str, Any], settings: Dict[str, Any]) -> Dict[str, Any]:
    """the base config with the settings put in; a nested setting changes only its own keys"""
    out = dict(base)
    for k, v in settings.items():
        out[k] = merged(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def same_rule(frozen: Dict[str, Any], current: Dict[str, Any]) -> bool:
    """does the card in the working tree say what the frozen card says? Only more seeds may be added (an INCONCLUSIVE verdict asks for one)"""
    strip = lambda c: {k: v for k, v in c.items() if k != "seeds"}      # noqa: E731
    return strip(frozen) == strip(current) and set(frozen.get("seeds", [])) <= set(current.get("seeds", [])) and \
        list(current.get("seeds", []))[:len(frozen.get("seeds", []))] == list(frozen.get("seeds", []))


def compose(base: Dict[str, Any], card: Dict[str, Any], arm: str, seed: int, rid: str) -> Dict[str, Any]:
    """the config of one run: the base config, the arm's settings, the seed and the run's place ({runs}: the machine's run root of the problem).
    `needs` become the folder of the named run of the same seed"""
    a = card["arms"][arm]
    cfg = merged(base, a.get("set", {}))
    for key, ref in a.get("needs", {}).items():
        exp, other = FROM.match(ref).groups()
        cfg[key] = f"{{runs}}/{exp}.{other}_s{seed}"
    for key, value in (("seed", seed), ("name", rid), ("out_root", "{runs}")):
        if key in base:
            cfg[key] = value
    return cfg
