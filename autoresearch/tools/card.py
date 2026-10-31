"""Metric cards: the evaluator contract (autoresearch/schemas/card.schema.json). Standard library only.

    python autoresearch/tools/card.py check <card.json> [...]        # exit 1 when a card breaks the contract
    python autoresearch/tools/card.py check --all [--strict]          # every card under the problem's card folder (project.json "cards", default runs/eval);
                                                                      # scratch cards (*_golden_check.json, *_selftest.json) are skipped
    python autoresearch/tools/card.py show <card.json>                # name, split, denominator, commit and the headline metrics

A card is what a problem's evaluator writes for one named system on one split; decide.py, experiment.py and the reports read cards only.
validate(card, headline, legacy_ok) returns the list of problems (empty = valid). A card written before the contract (no `schema` field but
otherwise complete) is a LEGACY card: accepted when legacy_ok (the default; `check --strict` turns it off), so committed records stay readable.
"""
import argparse
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import project as P  # noqa: E402

SCHEMA = os.path.join(P.AR, "schemas", "card.schema.json")
TYPES = {"str": str, "int": int, "bool": bool, "dict": dict, "list": list, "number": (int, float)}
SCRATCH = ("_golden_check.json", "_selftest.json")


def schema():
    return json.load(open(SCHEMA, encoding="utf-8"))


def is_number(v):
    """a finite number: NaN and infinity are not decision inputs"""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def is_legacy(card):
    """a full card from before the contract: everything but the schema field"""
    return isinstance(card, dict) and "schema" not in card and isinstance(card.get("metrics"), dict) and "name" in card


def validate(card, headline=(), legacy_ok=True):
    """problems of a card against the contract (empty list = valid)"""
    if not isinstance(card, dict):
        return ["not a JSON object"]
    sc = schema()
    legacy = is_legacy(card)
    bad = [f"missing field {k}" for k in sc["required"] if k not in card and not (legacy and legacy_ok and k in ("schema", "split", "commit", "date"))]
    for group in sc.get("one_of", []):
        if not any(k in card for k in group):
            bad.append("one of " + " / ".join(group) + " is needed (the denominator)")
    for k, t in sc["types"].items():
        if k in card and (not isinstance(card[k], TYPES[t]) or (t == "int" and isinstance(card[k], bool))):
            bad.append(f"{k} must be {t}")
    if bad:
        return bad
    if "schema" in card and card["schema"] != sc["schema"]:
        bad.append(f"schema {card['schema']} is not the contract's {sc['schema']}")
    if not str(card.get("name", "")).strip() or ("split" in card and not card["split"].strip()):
        bad.append("name and split must not be empty")
    for k, v in card["metrics"].items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            bad.append(f"metric {k} is not a number")
        elif not math.isfinite(v):
            bad.append(f"metric {k} is {'NaN' if v != v else 'infinite'}")
    if "seed" in card and (isinstance(card["seed"], bool) or not isinstance(card["seed"], (int, str))):
        bad.append("seed must be an int or a string (the training seed / repeat id that decide.py pairs the arms by)")
    if "protocol" in card and not isinstance(card["protocol"], (str, dict)):
        bad.append("protocol must be a string or a dict (what the evaluator says about its ground truth, cache and frame list)")
    n = card.get("frames", card.get("n"))
    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        bad.append("the denominator (frames / n) must be a positive integer")
    missing = [m for m in headline if m not in card["metrics"]]
    if missing:
        bad.append("headline metrics missing: " + ", ".join(missing))
    for where, blk in (("", card), ("compare.", card.get("compare") if isinstance(card.get("compare"), dict) else {})):
        bt = blk.get("better")
        if bt is None:
            continue
        if not isinstance(bt, dict) or any(v not in ("lower", "higher") for v in bt.values()):
            bad.append(f"{where}better must map metric names to 'lower' or 'higher'")
        elif any(k not in blk.get("metrics", {}) for k in bt):
            bad.append(f"{where}better names a metric the card does not hold: " + ", ".join(k for k in bt if k not in blk.get("metrics", {}))[:120])
    cmp = card.get("compare")
    if cmp is not None:
        if not isinstance(cmp.get("name"), str) or not isinstance(cmp.get("metrics"), dict):
            bad.append("compare needs name and metrics")
        elif any(not is_number(v) for v in cmp["metrics"].values()):
            bad.append("compare.metrics must be numbers")
        paired = cmp.get("paired", {})
        if not isinstance(paired, dict):
            bad.append("compare.paired must be a dict")
        else:
            for m, pr in paired.items():
                if not isinstance(pr, dict):
                    continue
                if "diff" in pr and not is_number(pr["diff"]):
                    bad.append(f"compare.paired.{m}: diff must be a finite number")
                elif "diff" in pr and not ("lo" in pr and "hi" in pr):
                    bad.append(f"compare.paired.{m} has diff but no lo / hi")
                elif "lo" in pr and "hi" in pr and (not is_number(pr["lo"]) or not is_number(pr["hi"]) or pr["lo"] > pr["hi"]):
                    bad.append(f"compare.paired.{m}: lo / hi must be numbers with lo <= hi")
    return bad


def load(path):
    return json.load(open(path, encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["check", "show"])
    ap.add_argument("cards", nargs="*")
    ap.add_argument("--all", action="store_true", help="every card under the problem's card folder (scratch cards skipped)")
    ap.add_argument("--strict", action="store_true", help="a legacy card (no schema field) counts as broken")
    a = ap.parse_intermixed_args()  # options may follow the card paths
    cfg = P.cfg()
    headline = cfg.get("headline_metrics", [])
    files = list(a.cards)
    if a.all:
        files += [f for f in sorted(glob.glob(os.path.join(P.root(), cfg.get("cards", "runs/eval"), "*", "*.json"))) if not f.endswith(SCRATCH)]
    if not files:
        ap.error("no cards given (paths or --all)")
    rc, n_legacy = 0, 0
    for f in files:
        try:
            card = load(f)
        except Exception as e:
            print(f"BAD    {f}: unreadable ({e})")
            rc = 1
            continue
        if a.cmd == "show":
            m = card.get("metrics", {})
            print(f"{card.get('name')} on {card.get('split')}{' ' + card['subset'] if card.get('subset') else ''}, n {card.get('frames', card.get('n'))}, commit {card.get('commit')}, {card.get('date')}: "
                  + "  ".join(f"{k} {m[k]:.4f}" for k in (headline or list(m)[:8]) if k in m))
            continue
        try:
            bad = validate(card, headline, legacy_ok=not a.strict)
        except Exception as e:
            bad = [f"validator error: {type(e).__name__}: {e}"]
        legacy = is_legacy(card) and not bad
        n_legacy += legacy
        print(f"{'LEGACY' if legacy else 'ok    ' if not bad else 'BAD   '} {f}" + "".join(f"\n    - {b}" for b in bad))
        rc |= bool(bad)
    if n_legacy:
        print(f"{n_legacy} legacy card(s) without the schema field (written before 2026-10-25); rewrite them with the evaluator when they matter")
    return rc


def selftest():
    good = dict(schema=1, name="v", split="val", frames=10, metrics=dict(Q=0.7, ms=12.5), commit="abc", date="2026-10-25 17:00")
    assert validate(good) == []
    assert validate({k: v for k, v in dict(good, n=3).items() if k != "frames"}) == []
    assert any("denominator" in b for b in validate({k: v for k, v in good.items() if k != "frames"}))
    assert any("missing field commit" in b for b in validate({k: v for k, v in good.items() if k != "commit"}))
    assert any("NaN" in b for b in validate(dict(good, metrics=dict(Q=float("nan")))))
    assert any("NaN" in b for b in validate(json.loads('{"schema": 1, "name": "v", "split": "val", "frames": 1, "metrics": {"Q": NaN}, "commit": "c", "date": "d"}')))
    assert any("infinite" in b for b in validate(dict(good, metrics=dict(Q=float("inf")))))
    assert any("must be numbers" in b for b in validate(dict(good, compare=dict(name="c", metrics=dict(Q=float("-inf"))))))
    assert any("lo / hi" in b for b in validate(dict(good, compare=dict(name="c", metrics={}, paired=dict(Q=dict(diff=0.1, lo=float("-inf"), hi=0.2))))))
    assert validate(dict(good, seed=1, protocol="gt-v2")) == [] and validate(dict(good, seed="s0", protocol=dict(gt="v2"))) == []
    assert any("seed must be" in b for b in validate(dict(good, seed=1.5))) and any("protocol must be" in b for b in validate(dict(good, protocol=3)))
    assert any("not a number" in b for b in validate(dict(good, metrics=dict(Q="0.7"))))
    assert any("headline" in b for b in validate(good, headline=["Q", "J"]))
    assert validate(dict(good, compare=dict(name="c", metrics=dict(Q=0.6), paired=dict(Q=dict(diff=0.1, lo=0.05, hi=0.15))))) == []
    assert any("lo / hi" in b for b in validate(dict(good, compare=dict(name="c", metrics={}, paired=dict(Q=dict(diff=0.1))))))
    assert any("lo <= hi" in b for b in validate(dict(good, compare=dict(name="c", metrics={}, paired=dict(Q=dict(diff=0.1, lo=0.2, hi=0.1))))))
    assert any("paired must be a dict" in b for b in validate(dict(good, compare=dict(name="c", metrics={}, paired=[1]))))
    assert validate([1, 2]) == ["not a JSON object"]
    legacy = {k: v for k, v in good.items() if k != "schema"}
    assert validate(legacy) == [] and is_legacy(legacy), "a card from before the contract is accepted"
    assert any("missing field schema" in b for b in validate(legacy, legacy_ok=False))
    assert any("frames must be int" in b for b in validate(dict(good, frames=True)))
    print("card selftest passed")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        sys.exit(main())
