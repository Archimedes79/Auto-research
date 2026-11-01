"""The experiment card (the machine-readable part of experiments/E<id>_<slug>/experiment.md, schema in autoresearch/schemas/experiment.schema.json). Standard library only.

The card is the first fenced block that starts with ```json card of the experiment.md; an older experiment.json next to it (or alone) is still read.
Both present: they must be equal (check says so), so a migration cannot leave two truths.

    python autoresearch/tools/experiment.py check E240          # gate G2 as code: fields, enums, arms, control, rule clauses, card refs resolve
    python autoresearch/tools/experiment.py show E240           # the arms with their cards and the headline numbers
    python autoresearch/tools/experiment.py check --all         # every experiment folder that has a card (block or experiment.json)
    python autoresearch/tools/experiment.py migrate E240        # card block into experiment.md, the json moves to <problem>/_to_delete/ (--all: every folder)

An experiment is found by its id (prefix of a folder under experiments/), by its folder or by the path of its experiment.md / experiment.json. Card refs: a path relative to the
problem folder, optionally '#a/b/c' into the JSON; a card has its metrics under 'metrics' (evaluate.py cards), a flat dict is the metrics itself.
"""
import argparse
import datetime
import glob
import json
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import project as P  # noqa: E402
from decide import card_lower, decide_experiment, named_lower  # noqa: E402
import card as CARD  # noqa: E402

SCHEMA = os.path.join(P.AR, "schemas", "experiment.schema.json")
TYPES = {"str": str, "number": (int, float), "dict": dict, "list": list, "bool": bool}


def schema():
    return json.load(open(SCHEMA, encoding="utf-8"))


FENCE = re.compile(r"^```json card[ \t]*\r?\n(\{.*?\})\r?\n```[ \t]*$", re.S | re.M)  # the body must be an object: an empty block or an unclosed one is not a card
OUTER = re.compile(r"^(`{4,})[^\n]*\n.*?^\1[ \t]*$", re.S | re.M)  # a sample of the format inside a longer fence (documentation) is prose, not a card


def bodies(text):
    """the raw json texts of the card blocks of a markdown text (blocks inside a fence of 4+ backticks do not count)"""
    return FENCE.findall(OUTER.sub("", text))


def block_of(md):
    """the card dict of an experiment.md, None when it has no ```json card block; ValueError for two blocks or a block that is not valid json"""
    if not os.path.isfile(md):
        return None
    found = bodies(open(md, encoding="utf-8").read())
    if not found:
        return None
    if len(found) > 1:
        raise ValueError(f"{md} has {len(found)} ```json card blocks, expected 1")
    try:
        return json.loads(found[0])
    except ValueError as e:
        raise ValueError(f"the card block of {md} is not valid json ({e})") from None


def source(folder):
    """the file that holds the card of an experiment folder: experiment.md with a card block, else experiment.json, else None"""
    md, js = os.path.join(folder, "experiment.md"), os.path.join(folder, "experiment.json")
    if os.path.isfile(md) and bodies(open(md, encoding="utf-8").read()):
        return md
    return js if os.path.isfile(js) else None


def find(exp):
    """path of the file that holds the card (experiment.md or experiment.json) for an id, a folder or a file"""
    if os.path.isfile(exp):
        return exp
    if os.path.isdir(exp):
        return source(exp) or os.path.join(exp, "experiment.json")
    cand = sorted(glob.glob(os.path.join(P.root(), "experiments", exp + "_*")) + glob.glob(os.path.join(P.root(), "experiments", exp)))
    hits = [h for h in (source(d) for d in cand if os.path.isdir(d)) if h]
    if not hits:
        sys.exit(f"no card (experiment.md with a ```json card block, or experiment.json) for {exp} under {P.root()}/experiments")
    return hits[0]


def read(path):
    """the card dict of an experiment.md (block) or an experiment.json; ValueError when it is missing, unreadable or not valid json"""
    if path.endswith(".md"):
        card = block_of(path)
        if card is None:
            raise ValueError(f"{path} has no ```json card block")
        return card
    try:
        return json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ValueError(f"cannot read {path}: {e}") from None


def load(exp):
    p = find(exp)
    try:
        return read(p), p
    except ValueError as e:
        sys.exit(str(e))


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
    bad = []
    folder = os.path.dirname(os.path.abspath(path))
    js, md = os.path.join(folder, "experiment.json"), os.path.join(folder, "experiment.md")
    if os.path.abspath(path) in (js, md) and os.path.isfile(js) and os.path.isfile(md):
        try:
            if block_of(md) is not None and json.load(open(js, encoding="utf-8")) != block_of(md):
                bad.append("experiment.json and the card block of experiment.md differ (two truths: keep one, `experiment.py migrate` moves the json away)")
        except ValueError as e:
            bad.append(f"cannot compare experiment.json with the card block: {e}")
    bad += [f"missing field {k}" for k in sc["required"] if k not in exp]
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
    elif variant == exp["control"]:
        bad.append("the variant arm is the control")
    clauses = [c for c in exp["rule"] if isinstance(c, dict)]
    typed = [f"arm {a} must be an object" for a, arm in arms.items() if not isinstance(arm, dict)]  # hand-typing errors are reported, never raised
    for a, arm in arms.items():
        if isinstance(arm, dict) and "names" in arm and not (isinstance(arm["names"], list) and all(isinstance(n, str) for n in arm["names"])):
            typed.append(f"arm {a}: names must be a list of strings")
        if isinstance(arm, dict) and "cards" in arm and not isinstance(arm["cards"], (list, dict)):
            typed.append(f"arm {a}: cards must be a list or an object")
    if not isinstance(exp["prediction"].get("vs"), str) or any(not isinstance(c.get("metric", ""), str) or not isinstance(c.get("vs", ""), str) for c in clauses):
        typed.append("prediction.vs, and the metric and vs of every clause, must be strings (arm letters, metric names)")
    if typed:
        return bad + typed
    for a, arm in arms.items():
        if not any(k in arm for k in ("names", "cards", "recipe", "command")):
            bad.append(f"arm {a}: needs names / cards (results) or recipe / command (how to run)")
    pred = exp["prediction"]
    if not isinstance(pred.get("value"), (int, float)) or pred.get("vs") not in arms:
        bad.append("prediction needs {value: number, vs: <arm letter>}")
    thresholds = [("band", exp["band"]), ("prediction.value", pred.get("value"))] + [
        (f"clause {i} {k}", c[k]) for i, c in enumerate(exp["rule"]) if isinstance(c, dict) for k in ("min_effect", "max_loss", "band") if k in c]
    bad += [f"{name} must be a finite number" for name, v in thresholds if not CARD.is_number(v)]
    allow = exp["evaluate"].get("allow", [])
    if not isinstance(allow, list) or any(x not in ("split", "subset", "denominator", "protocol", "provenance") for x in allow):
        bad.append("evaluate.allow must be a list of: split, subset, denominator, protocol, provenance")
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
    resolved = 0  # arms whose cards exist: with all of them the decision itself is tried below
    for a in arms:
        try:
            cards = arm_cards(exp, a, root)
        except (FileNotFoundError, KeyError, ValueError) as e:
            bad.append(f"arm {a}: {e}")
            continue
        except Exception as e:  # noqa: BLE001  (a card ref of the wrong type, ...: reported, never raised)
            bad.append(f"arm {a}: cannot read its cards ({type(e).__name__}: {e})")
            continue
        resolved += bool(cards)
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
    if not bad and resolved == len(arms):  # every card is there and sound: the decision must not refuse them (same split / frames, seeds that pair)
        try:
            decide_experiment(exp, root)
        except ValueError as e:
            bad.append(f"cannot be decided: {e}")
        except (KeyError, TypeError, AttributeError, OverflowError, StopIteration) as e:  # a malformed file is reported, check never raises
            bad.append(f"cannot be decided: {type(e).__name__}: {e}")
    return bad


def migrate(folder, root):
    """card block into experiment.md (right after the first heading outside a fence, line endings kept), the json moves to <problem>/_to_delete/experiment_json_<date>/<folder>/ (never deleted); -> message"""
    js, md = os.path.join(folder, "experiment.json"), os.path.join(folder, "experiment.md")
    name = os.path.basename(folder)
    if not os.path.isfile(md):
        return f"{name}: JSON-ONLY (no experiment.md to hold the block): left as it is"
    try:
        card = read(js)
        have = block_of(md)
    except ValueError as e:
        return f"{name}: NOT migrated, {e}"
    if have is not None:
        if have != card:
            return f"{name}: the block and the json differ: NOT migrated (decide which is right)"
    else:
        text = open(md, encoding="utf-8", newline="").read()
        nl = "\r\n" if "\r\n" in text else "\n"
        lines = text.split(nl)
        i, fence = 0, False
        for k, ln in enumerate(lines):  # after the first heading that is not inside a code fence, else at the top
            if ln.lstrip().startswith("```"):
                fence = not fence
            elif not fence and ln.startswith("# "):
                i = k + 1
                break
        block = ["```json card"] + json.dumps(card, indent=1, ensure_ascii=False).split("\n") + ["```"]
        new = nl.join(lines[:i] + ["", "The card (read by `experiment.py` and `decide.py`; schema `autoresearch/schemas/experiment.schema.json`):", ""] + block + [""] + lines[i:])
        tmp = md + ".new"
        open(tmp, "w", encoding="utf-8", newline="").write(new)
        try:
            ok = block_of(tmp) == card
        except ValueError:
            ok = False
        if not ok:  # never replace a file with one that does not read back
            os.remove(tmp)
            return f"{name}: ERROR the block does not read back equal: NOT migrated"
        os.replace(tmp, md)
    dest = os.path.join(root, "_to_delete", "experiment_json_" + datetime.date.today().isoformat(), name)
    os.makedirs(dest, exist_ok=True)
    target, n = os.path.join(dest, "experiment.json"), 1
    while os.path.exists(target):  # never overwrite an earlier move
        n += 1
        target = os.path.join(dest, f"experiment.{n}.json")
    shutil.move(js, target)
    return f"{name}: migrated (json in {os.path.relpath(dest, root)})"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["check", "show", "migrate"])
    ap.add_argument("exp", nargs="?")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if not a.all and not a.exp:
        ap.error("name an experiment (E240, a folder or a file) or pass --all")
    if a.cmd == "migrate":
        todo = [d for d in sorted(glob.glob(os.path.join(P.root(), "experiments", "*"))) if os.path.isdir(d)] if a.all else [os.path.dirname(find(a.exp))]
        for d in todo:
            if os.path.isfile(os.path.join(d, "experiment.json")):
                print(migrate(d, P.root()))
        return 0
    files = [source(d) for d in sorted(glob.glob(os.path.join(P.root(), "experiments", "*"))) if os.path.isdir(d) and source(d)] if a.all else [find(a.exp)]
    rc = 0
    for f in files:
        try:
            exp = read(f)
        except ValueError as e:  # a broken card is reported and the other experiments are still checked
            print(f"?        NOT READY  {os.path.relpath(f, P.root())}\n    - {e}")
            rc = 1
            continue
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
