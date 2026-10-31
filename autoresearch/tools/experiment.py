"""The experiment file (experiments/E<id>_<slug>/experiment.json, schema in autoresearch/schemas/experiment.schema.json). Standard library only.

    python autoresearch/tools/experiment.py check E240          # gate G2 as code: fields, enums, arms, control, rule clauses, card refs resolve
    python autoresearch/tools/experiment.py show E240           # the arms with their cards and the headline numbers
    python autoresearch/tools/experiment.py check --all         # every experiment folder that has an experiment.json

An experiment is found by its id (prefix of a folder under experiments/) or by the path of its experiment.json. Card refs: a path relative to the
problem folder, optionally '#a/b/c' into the JSON; a card has its metrics under 'metrics' (evaluate.py cards), a flat dict is the metrics itself.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import project as P  # noqa: E402
from decide import card_lower, named_lower  # noqa: E402
import card as CARD  # noqa: E402

SCHEMA = os.path.join(P.AR, "schemas", "experiment.schema.json")
TYPES = {"str": str, "number": (int, float), "dict": dict, "list": list, "bool": bool}


def schema():
    return json.load(open(SCHEMA, encoding="utf-8"))


def find(exp):
    """path of the experiment.json for an id, a folder or a file"""
    if os.path.isfile(exp):
        return exp
    if os.path.isdir(exp):
        return os.path.join(exp, "experiment.json")
    hits = sorted(glob.glob(os.path.join(P.root(), "experiments", exp + "_*", "experiment.json"))) + sorted(glob.glob(os.path.join(P.root(), "experiments", exp, "experiment.json")))
    if not hits:
        sys.exit(f"no experiments/{exp}_*/experiment.json under {P.root()}")
    return hits[0]


def load(exp):
    p = find(exp)
    return json.load(open(p, encoding="utf-8")), p


def card_ref(ref, root=None):
    """(metrics dict, card dict or None, resolved description) of a card ref"""
    root = root or P.root()
    path, _, inner = ref.partition("#")
    full = path if os.path.isabs(path) else os.path.join(root, path)
    if not os.path.exists(full):
        raise FileNotFoundError(f"card {path} not found")
    d = json.load(open(full, encoding="utf-8"))
    for k in [x for x in inner.split("/") if x]:
        if not isinstance(d, dict) or k not in d:
            raise KeyError(f"{ref}: key {k} not in the JSON")
        d = d[k]
    if not isinstance(d, dict):
        raise ValueError(f"{ref}: not a dict")
    if "metrics" in d and isinstance(d["metrics"], dict):
        return d["metrics"], d, ref
    return d, None, ref


def arm_cards(exp, letter, root=None):
    """{subset or '': [(metrics, card, ref), ...]} for an arm; names become evaluate.py cards of the experiment's split / subset"""
    root = root or P.root()
    arm = exp["arms"][letter]
    ev = exp.get("evaluate", {})
    split, sub = ev.get("split", "val"), ev.get("subset", "")
    out = {}
    if "names" in arm:
        folder = P.cfg_of(root).get("cards", "runs/eval")  # where this problem's evaluator writes its cards
        out[""] = [card_ref(f"{folder}/{n}/{split}{'_' + sub if sub else ''}.json", root) for n in arm["names"]]
    cards = arm.get("cards")
    if isinstance(cards, list):
        out[""] = out.get("", []) + [card_ref(c, root) for c in cards]
    elif isinstance(cards, dict):
        for s, refs in cards.items():
            out[s] = [card_ref(c, root) for c in (refs if isinstance(refs, list) else [refs])]
    return out


def check(exp, path):
    """list of problems (empty = ready)"""
    sc = schema()
    bad = [f"missing field {k}" for k in sc["required"] if k not in exp]
    for k, t in sc["types"].items():
        if k in exp and not isinstance(exp[k], TYPES[t]):
            bad.append(f"{k} must be {t}")
    for k, allowed in sc["enum"].items():
        if k in exp and exp[k] not in allowed:
            bad.append(f"{k} must be one of {allowed}")
    if bad:
        return bad
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(path)))) if "experiments" in path.replace("\\", "/") else P.root()
    arms = exp["arms"]
    if exp["control"] not in arms:
        bad.append(f"control {exp['control']} is not an arm")
    variant = exp.get("variant") or next((a for a in arms if a != exp["control"]), None)
    if variant is None or variant not in arms:
        bad.append("no variant arm (an arm other than the control)")
    for a, arm in arms.items():
        if not any(k in arm for k in ("names", "cards", "recipe", "command")):
            bad.append(f"arm {a}: needs names / cards (results) or recipe / command (how to run)")
    pred = exp["prediction"]
    if not isinstance(pred.get("value"), (int, float)) or pred.get("vs") not in arms:
        bad.append("prediction needs {value: number, vs: <arm letter>}")
    if not exp["rule"]:
        bad.append("rule has no clause")
    for i, c in enumerate(exp["rule"]):
        if not isinstance(c, dict) or "metric" not in c:
            bad.append(f"clause {i}: needs a metric")
            continue
        if ("min_effect" in c) == ("max_loss" in c):
            bad.append(f"clause {i} ({c['metric']}): exactly one of min_effect / max_loss")
        if c.get("vs", exp["control"]) not in arms:
            bad.append(f"clause {i}: vs {c.get('vs')} is not an arm")
        if c.get("subsets") and not isinstance(c["subsets"], list):
            bad.append(f"clause {i}: subsets must be a list")
    if not any(c["metric"] == exp["metric"] for c in exp["rule"] if isinstance(c, dict)):
        bad.append(f"no clause on the headline metric {exp['metric']}")
    if not isinstance(exp["evaluate"].get("split"), str) or not exp["evaluate"]["split"].strip():
        bad.append("evaluate.split must name the split or benchmark set (val | test | confirm | <benchmark>)")
    every = []  # the cards of all arms: the directions they declare must agree (decide.card_lower)
    for a in arms:
        try:
            cards = arm_cards(exp, a, root)
        except (FileNotFoundError, KeyError, ValueError) as e:
            bad.append(f"arm {a}: {e}")
            continue
        for s, lst in cards.items():
            for m, full, ref in lst:
                every.append((m, full, ref))
                if full is not None:  # a full card follows the evaluator contract (legacy cards without the schema field pass); a bare metrics dict is accepted as is
                    bad += [f"arm {a}: card {ref}: {b}" for b in CARD.validate(full, P.cfg_of(root).get("headline_metrics", []))]
                for c in exp["rule"]:
                    if isinstance(c, dict) and c.get("metric") and c["metric"] not in m and (not c.get("subsets") or s in c["subsets"]):
                        bad.append(f"arm {a}{' ' + s if s else ''}: card {ref} has no metric {c['metric']}")
    for metric in dict.fromkeys([exp["metric"]] + [c["metric"] for c in exp["rule"] if isinstance(c, dict) and c.get("metric")]):
        try:
            said = card_lower(metric, every, root)
        except ValueError as e:
            bad.append(str(e))
            continue
        if metric != exp["metric"]:
            continue
        if said is not None and said != (exp["direction"] == "lower"):
            bad.append(f"direction {exp['direction']} contradicts the cards: their evaluator declares {metric} {'lower' if said else 'higher'}-is-better")
        elif said is None and named_lower(metric, root) and exp["direction"] != "lower":
            bad.append(f"metric {metric} is lower-is-better (project.json lower_is_better) but direction is {exp['direction']}")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["check", "show"])
    ap.add_argument("exp", nargs="?")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(P.root(), "experiments", "E*", "experiment.json"))) if a.all else [find(a.exp)]
    rc = 0
    for f in files:
        exp = json.load(open(f, encoding="utf-8"))
        if a.cmd == "check":
            bad = check(exp, f)
            print(f"{exp.get('id', '?'):8} {'ready' if not bad else 'NOT READY'}  {os.path.relpath(f, P.root())}" + "".join(f"\n    - {b}" for b in bad))
            rc |= bool(bad)
        else:
            print(f"{exp['id']} {exp['title']} [{exp['kind']}] headline {exp['metric']} ({exp['direction']}, band {exp['band']}), control {exp['control']}")
            for letter in exp["arms"]:
                for s, lst in arm_cards(exp, letter).items():
                    for m, card, ref in lst:
                        print(f"  {letter}{' ' + s if s else '':12} {exp['metric']} {m.get(exp['metric'], float('nan')):.4f}  {ref}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
