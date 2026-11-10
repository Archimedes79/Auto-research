"""Decision rule as code: KEEP / CUT / TIE / INCONCLUSIVE, from an experiment file and its metric cards, or from typed per-seed scores.

    python decide.py --experiment E240 [--write] [--by-hand "reason"]   the rule clauses of the card of experiments/E240_*/experiment.md (an old experiment.json also works) applied to the cards
    python decide.py --variant 0.746,0.741 --control 0.725,0.722 [--band 0.010] [--lower-better] [--cost-variant 31 --cost-control 628] [--id E123.D7 --write]

Rule (autoresearch/METHOD.md, P3 step 5; schema autoresearch/schemas/experiment.schema.json):
  * effect = mean(variant) - mean(control) on the headline metric; with equal seed counts the seeds are paired (pair_arms): "pairing": "position"
    in the experiment = card i against card i; else BY SEED IDENTITY (the card's `seed`, else the arm's own `seeds` list; a seed only one arm
    ran or a duplicate is refused); else the experiment-level `seeds` list declares one common order of the lists (= position); else two or more
    seeds are refused. The decision records how (pairing: seed | position | single | unpaired). The typed interface pairs by position. Values
    must be finite numbers
  * the cards of the arms the rule uses must be measurements of the same thing (`incompatible`: split = evaluate.split, subset, denominator,
    protocol, a field missing on one card counts as different; evaluate.allow lists a known difference), else nothing is decided. Cards with a
    `provenance` block (tools/provenance.py) are compared in GT version, GT signature, evaluator code hash, frame hash and raw-data manifest hash, and a card whose evaluator
    differs from its commit is not used (`provenance_check`; allow "provenance"); a value a card does not know is named in the verdict as 'cannot tell'
  * |effect| <= band                                  -> TIE: take the cheaper arm
  * beyond the band in the worse direction              -> CUT
  * beyond the band in the better direction, but one seed only, a paired seed on the other side, or the paired scene-bootstrap interval
    (compare block of the card, when the card was made with evaluate.py --compare <control>) contains 0 -> INCONCLUSIVE: add a seed, do not adopt
  * beyond the band in the better direction with agreeing seeds (and interval) -> KEEP, if every clause of the rule holds:
      min_effect clause: variant - vs >= min_effect (with agreeing seeds); max_loss clause (a guard): variant - vs >= -max_loss;
      a clause with subsets passes when at_least (default: all) of its subsets pass. A violated guard -> CUT; a missed min_effect -> CUT ("rule not met")
  * deterministic: true (the arms share the checkpoint, e.g. a post-processing change): no training seed exists, so a single pair with an
    interval that excludes 0 counts as agreeing seeds
  * bands: the experiment's band for its headline metric, a clause's own band, else 0.010; lower-is-better: the experiment's direction for its
    headline metric, else the cards' own "better" map (the evaluator declares every metric it writes; cards that disagree are refused), else
    for older cards the generic cost names (LOWER) plus the problem's project.json "lower_is_better" list (a path-like card key such as
    'det/dt+0.00/vehicle/all/ate' is matched by its last part)
  * the headline is judged for its evidence (one seed, a disagreeing seed, an interval containing 0) also when its clause is a guard (max_loss)
    or it has no min_effect clause: a better headline on one seed is INCONCLUSIVE, not KEEP
  * the gate (hard rule 4: the rule is written before the run): every run id the variant and the control arm list needs a ledger row with status ok, and a run whose
    record carries the `card_sha` that run.py stored at launch must still match experiment.rule_sha(card); else nothing is decided ("the rule changed after run X was
    launched"). No run with a card_sha: decided, labelled "rule not registered at launch". --by-hand "reason" bypasses the gate and labels the verdict "by hand: reason"
    (decision.json, the ledger token, results.md). The typed form needs the ledger row of --id with status ok and is always "by hand: typed scores"
  * --write records, in this order: the decision column of the variant arm's ledger rows (run_ids), TOKEN FIRST: "decide.py <date>: VERDICT (effect ...)", then " | " and the
    prose that was there (cut to 400 characters in all; the token is never cut; an earlier token is replaced); decision.json next to the card (written atomically; the earlier
    verdicts stay in its `previous` list, 10 at most; it carries rule_sha); then result.py writes it where people look for it (results.md next to the card, the `result` column of
    the schedule, the research notes of `sources`: see the top of result.py; `python result.py E240` repeats that part alone).
    A ledger or decision.json failure stops with exit 1; the other records are side records: a failure is a printed warning, the exit code stays 0
This is the pre-registered decision, not a significance test: intervals come from the problem's evaluation (paired scene bootstrap).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from card import is_number  # noqa: E402  (a finite number: NaN, infinity and text are no decision inputs)

LOWER = {"ms", "ms_per_frame"}  # generic cost names; problem metrics are listed in project.json "lower_is_better"
DEFAULT_BAND = 0.010
DECIMALS = 9   # an effect is rounded to this many decimals before it meets a band or a threshold: 0.746 - 0.736 is 0.010000000000000009 in floats, and the rule says |effect| <= band is a tie
VERDICTS = "KEEP|CUT|TIE|INCONCLUSIVE"
TOKEN = re.compile(rf"decide\.py [\d-]+: ({VERDICTS})\b[^|]*")  # what this tool writes into the decision cell of a ledger row; it ends at the next ' | '
DECISION_CAP = 400   # characters of the decision cell: the token is never cut, the prose behind it is


def decide(variant, control, band=0.010, lower_better=False, cost_variant=None, cost_control=None):
    """typed scores, one per seed (the loop-18 interface): the position IS the pairing, seed i of the variant against seed i of the control
    (experiment files pair by seed identity instead: pair_arms)"""
    if not variant or not control or not all(is_number(x) for x in [*variant, *control, band]):
        raise ValueError("scores and band must be finite numbers, at least one score per arm")
    sgn = -1.0 if lower_better else 1.0
    mv, mc = sum(variant) / len(variant), sum(control) / len(control)
    effect = round(sgn * (mv - mc), DECIMALS)
    if len(variant) == len(control):
        diffs = [sgn * (v - c) for v, c in zip(variant, control)]
    else:
        diffs = [effect]
    if abs(effect) <= band:
        if cost_variant is not None and cost_control is not None:
            cheaper = "variant" if cost_variant < cost_control else "control"
            return "TIE", effect, f"inside the noise band {band}: take the cheaper one ({cheaper}: {cost_variant} vs {cost_control})"
        return "TIE", effect, f"inside the noise band {band}: take the simpler / cheaper one"
    if effect < 0:                                                    # worse beyond the band: CUT, as decide_experiment does (no seed can rescue it)
        return "CUT", effect, f"beyond the band {band} in the worse direction"
    same = all(d > 0 for d in diffs)
    if len(diffs) < 2 or not same:
        return "INCONCLUSIVE", effect, "beyond the band but " + ("only one seed" if len(diffs) < 2 else "a seed disagrees") + ": add a seed, do not adopt yet"
    return "KEEP", effect, f"beyond the band {band} on all {len(diffs)} paired seeds"


# ------------------------------------------------------------------ the decision cell of a ledger row: written here, read by keep3.py, metrics.py and plan.py


def decision_token(verdict, date, effect, notes=()):
    """the token for the decision cell: `decide.py <date>: KEEP (effect +0.0123, <note>, ...)`; TOKEN finds it wherever it stands in the cell"""
    notes = [" ".join(str(n).replace("|", "/").split())[:120] for n in notes]
    return f"decide.py {date}: {verdict} (effect {effect:+.4f}" + "".join(f", {n}" for n in notes) + ")"


def split_decision(text):
    """(prose, token) of a decision cell: the decide.py token wherever it stands, and what was written by hand around it. A cell without a dated token (older rows,
    'decide.py: ...' typed by hand) is all prose up to a ' | decide.py'"""
    text = text or ""
    m = TOKEN.search(text)
    if not m:
        return text.split(" | decide.py")[0].strip(), ""
    return (text[:m.start()].rstrip(" |") + " " + text[m.end():].lstrip(" |")).strip(), m.group(0).strip()


def verdict_of(text):
    """the verdict of a decision cell: that of the (last) decide.py token, in either order with the prose; a cell without a token is read by its last verdict word"""
    found = [m.group(1) for m in TOKEN.finditer(text or "")]
    return found[-1] if found else (re.findall(rf"\b({VERDICTS})\b", text or "") or [""])[-1]


def ledger_decision(old, token, cap=DECISION_CAP):
    """the new decision cell: the token FIRST, then ' | ' and the prose of the old cell (an earlier token goes); only the prose is cut to fit the cap, never the token"""
    prose = split_decision(old)[0]
    if prose.startswith("decide.py"):  # an older verdict typed by hand without the dated token: replaced, not kept as prose
        prose = ""
    room = cap - len(token) - 3
    return token + (" | " + prose[:room].rstrip() if prose and room > 0 else "")


# ------------------------------------------------------------------ experiment file + cards


def lower_names(root=None):
    """the lower-is-better metric names of a problem: project.json "lower_is_better" plus the generic LOWER"""
    import project as P
    try:
        c = P.cfg_of(root or P.root())
    except SystemExit:  # no problem folder (typed interface outside a project)
        return set(LOWER)
    return LOWER | set(c.get("lower_is_better", []))


def named_lower(metric, root=None):
    """the fallback for a metric no card declares: lower is better when lower_names lists the card key or its last path part
    (a path-like key 'det/dt+0.00/vehicle/all/ate' is listed as 'ate')"""
    names = lower_names(root)
    return metric in names or metric.rsplit("/", 1)[-1] in names


def card_lower(metric, cards=(), root=None):
    """the direction the cards declare for a metric: True (lower is better) / False, or None when no card declares it. cards: (metrics,
    card, ref) tuples; only cards that HOLD the metric count. A card declares through its "better" map ({metric: "lower" | "higher"},
    written by its evaluator for every metric it writes, card contract); a card that holds the metric without declaring it (written
    before the field, or a metric a script added) is undeclared and is read by the fallback named_lower. Refused (ValueError): declaring
    cards that disagree, and a declaration that contradicts how the undeclared cards of the same comparison would be read"""
    said, undeclared = {}, []
    for m, c, ref in cards:
        if not isinstance(m, dict) or metric not in m:
            continue
        b = c.get("better") if isinstance(c, dict) else None
        if isinstance(b, dict) and b.get(metric) in ("lower", "higher"):
            said.setdefault(b[metric] == "lower", []).append(str(ref))
        else:
            undeclared.append(str(ref))
    if len(said) > 1:
        raise ValueError(f"the cards disagree whether {metric} is lower-is-better (lower: {', '.join(said[True])}; higher: {', '.join(said[False])})")
    if not said:
        return None
    low = next(iter(said))
    if undeclared and low != named_lower(metric, root):
        raise ValueError(f"{metric}: {', '.join(said[low])} declare it {'lower' if low else 'higher'}-is-better, but the older cards "
                         f"{', '.join(undeclared)} would be read the other way (fallback: project.json lower_is_better); fix the fallback")
    return low


def is_lower(metric, exp=None, root=None, cards=()):
    """lower is better: the experiment's direction for its headline metric, else what the cards declare (card_lower), else the
    problem's names (named_lower: project.json, the generic LOWER) for cards written before the field existed"""
    if exp is not None and metric == exp.get("metric"):
        return exp.get("direction") == "lower"
    declared = card_lower(metric, cards, root)
    return declared if declared is not None else named_lower(metric, root)


def paired_effect(metric, v_cards, c_cards, lower=None):
    """(effect, paired diffs, interval or None) of a metric between two lists of (metrics, card, ref); the sign is 'higher is better'.
    lower: the direction (pass it: the default reads the AR_PROJECT / default problem, not necessarily the experiment's)"""
    sgn = -1.0 if (is_lower(metric) if lower is None else lower) else 1.0
    v = [m[metric] for m, _, _ in v_cards]
    c = [m[metric] for m, _, _ in c_cards]
    if not v or not c:
        raise KeyError(f"no card for {metric}")
    if not all(is_number(x) for x in [*v, *c]):
        raise ValueError(f"{metric}: a card holds a value that is not a finite number (NaN, infinity or text)")
    effect = round(sgn * (sum(v) / len(v) - sum(c) / len(c)), DECIMALS)
    diffs = [sgn * (a - b) for a, b in zip(v, c)] if len(v) == len(c) else [effect]
    spans = []
    c_names = {(cc or {}).get("name") for _, cc, _ in c_cards}
    for m, card, _ in v_cards:  # the paired scene bootstrap of evaluate.py --compare, of every card that compared against this control
        cmp = (card or {}).get("compare") or {}
        pr = cmp.get("paired", {}).get(metric)
        if isinstance(pr, dict) and "lo" in pr and cmp.get("name") in c_names:
            lo, hi = sgn * pr["lo"], sgn * pr["hi"]
            spans.append((min(lo, hi), max(lo, hi)))
    interval = (min(s[0] for s in spans), max(s[1] for s in spans)) if spans else None    # the hull: one seed whose interval holds 0 flags it
    return effect, diffs, interval


def seed_ids(exp, a, cards, key=""):
    """the seed identity of each card of arm a, as strings in card order, or None when unknown: the card's own `seed`, else the arm's own `seeds`
    list. The experiment-level `seeds` is NOT an identity (it says which seeds the arms run, not in which order their cards are listed: see
    pair_arms). An arm `seeds` list of the wrong length for the whole-split cards (key "") is an error"""
    own = [(c or {}).get("seed") for _, c, _ in cards]
    if cards and all(s is not None for s in own):
        return [str(s) for s in own]
    declared = exp["arms"][a].get("seeds")
    if isinstance(declared, list) and cards:
        if len(declared) == len(cards):
            return [str(s) for s in declared]
        if key == "":
            raise ValueError(f"arm {a} lists {len(declared)} seeds for {len(cards)} cards")
    return None


def pair_arms(exp, a, b, cards_a, cards_b, key=""):
    """(cards_a, cards_b in the order of cards_a, how): the cards of arms a and b paired seed by seed. Unequal counts are not paired (how =
    "unpaired": the caller counts them as one seed); fewer than two cards are one pair ("single"). Otherwise: "pairing": "position" in the
    experiment pairs card i with card i ("position", the lists' order is the pairing); else by seed identity (seed_ids; a duplicate or a seed
    only one arm ran is a ValueError) ("seed"); else the experiment-level `seeds` list, as long as the cards, declares one common order of the
    lists ("position"); else ValueError (nothing says which cards belong together)."""
    if len(cards_a) != len(cards_b):
        return cards_a, cards_b, "unpaired"
    if len(cards_a) < 2:
        return cards_a, cards_b, "single"
    if exp.get("pairing") == "position":
        return cards_a, cards_b, "position"
    ids_a, ids_b = seed_ids(exp, a, cards_a, key), seed_ids(exp, b, cards_b, key)
    if ids_a is not None and ids_b is not None:
        for arm, ids in ((a, ids_a), (b, ids_b)):
            if len(set(ids)) != len(ids):
                raise ValueError(f"arm {arm} lists a seed twice ({', '.join(ids)}): the arms cannot be paired")
        if set(ids_a) != set(ids_b):
            raise ValueError(f"the seeds of arm {a} ({', '.join(ids_a)}) and arm {b} ({', '.join(ids_b)}) are not the same: a seed that only one arm ran has no pair")
        by_seed = dict(zip(ids_b, cards_b))
        return cards_a, [by_seed[s] for s in ids_a], "seed"
    if isinstance(exp.get("seeds"), list) and len(exp["seeds"]) == len(cards_a):
        return cards_a, cards_b, "position"
    raise ValueError(f"arms {a} and {b} have {len(cards_a)} cards each but no seed identities to pair them by: give every card a `seed`, or each arm a `seeds` "
                     "list as long as its cards, or declare the common order with a `seeds` list of the experiment, or \"pairing\": \"position\" (card i against card i)")


def incompatible(exp, cards):
    """the reasons why the cards of the arms are not measurements of the same thing, from the cards' own fields: split (must be the experiment's
    evaluate.split), subset (of whole-split cards, when the card has the field: evaluate.subset), the denominator (frames / n) and the optional
    protocol (what the evaluator says about ground truth, cache, frame list) must be the same on every card of one subset key; a card that lacks
    a field that another card has counts as different (a protocol that is a dict is compared key by key: a key that only some cards carry is
    unknown on the others, not a difference). Only the arms the rule uses are compared (the variant, the control, the clauses' `vs`).
    A bare metrics dict holds no fields and passes. exp["evaluate"]["allow"] lists fields that are knowingly different (split | subset |
    denominator | protocol)."""
    ev = exp.get("evaluate", {})
    allow = ev.get("allow") or []
    if not isinstance(allow, list):
        raise ValueError("evaluate.allow must be a list of fields (split, subset, denominator, protocol, provenance)")
    allow = set(allow)
    used = used_arms(exp)
    out, seen = [], {}
    for a in sorted(used):
        for s, lst in cards[a].items():
            for _, card, ref in lst:
                if not card:
                    continue
                if card.get("split") is not None and "split" not in allow and card["split"] != ev.get("split"):
                    out.append(f"{ref}: split {card['split']!r}, the experiment evaluates {ev.get('split')!r}")
                if s == "" and "subset" in card and "subset" not in allow and (card["subset"] or "") != ev.get("subset", ""):
                    out.append(f"{ref}: subset {card['subset']!r}, the experiment evaluates {ev.get('subset', '')!r}")
                for field, value in (("denominator", card.get("frames", card.get("n"))), ("protocol", card.get("protocol"))):
                    if field not in allow:
                        seen.setdefault((s, field), {}).setdefault(json.dumps(value, sort_keys=True), []).append(str(ref))
    for (s, field), values in sorted(seen.items()):
        parsed = {v: json.loads(v) for v in values}
        if field == "protocol" and len(values) > 1 and all(isinstance(p, dict) for p in parsed.values()):
            for key in sorted({k for p in parsed.values() for k in p}):          # a protocol that grew: a key only some cards carry is unknown on the others, not a difference
                by_value = {}
                for v, p in parsed.items():
                    if key in p:
                        by_value.setdefault(json.dumps(p[key], sort_keys=True), []).extend(values[v])
                if len(by_value) > 1:
                    out.append(f"{'subset ' + s + ': ' if s else ''}the cards differ in protocol.{key}: " + "; ".join(
                        f"{json.loads(v)!r} ({', '.join(refs[:2])})" for v, refs in by_value.items()))
        elif len(values) > 1:
            out.append(f"{'subset ' + s + ': ' if s else ''}the cards differ in {field}: " + "; ".join(
                f"{'missing' if v == 'null' else json.loads(v)!r} ({', '.join(refs[:2])})" for v, refs in values.items()))
    if "provenance" not in allow:
        out += provenance_check(exp, cards)[0]
    return out


def used_arms(exp):
    """the arms a decision compares: the variant, the control and the `vs` of every clause"""
    variant = exp.get("variant") or next(a for a in exp["arms"] if a != exp["control"])
    return {variant, exp["control"], *[c.get("vs", exp["control"]) for c in exp["rule"] if isinstance(c, dict)]}


def _prov_value(p, key):
    """one value of a provenance block (None when the block or the value is missing)"""
    for k in key:
        p = p.get(k) if isinstance(p, dict) else None
    return p


def provenance_check(exp, cards):
    """(differences, cannot_tell) over the `provenance` blocks (tools/provenance.py, autoresearch/reproducibility.md) of the cards one decision compares, per subset
    key and over ALL cards of the group (not each against the first: an unknown value on one card must not hide a mismatch between the others). Per key of
    provenance.COMPARE_KEYS (GT version, GT signature, evaluator code hash, frame hash, raw-data manifest hash): two or more distinct known values are a
    difference (it blocks the comparison); a card that does not know the value (a missing block, 'unknown') is listed as 'cannot tell', never a refusal. The state of
    the evaluator is a blocker of its own, once per card: it differs from its commit (`evaluator.dirty` lists files), or it was not checked (anything that is no
    list: 'unknown', missing). Nothing at all when no card in the group carries a block (the feature is not in use yet: an old card keeps what it has).
    evaluate.allow ["provenance"] waives the differences (decide_experiment records what was waived)"""
    import provenance as PV
    groups = {}
    for a in sorted(used_arms(exp)):
        for s, lst in cards[a].items():
            for _, card, ref in lst:
                if card:
                    p = card.get("provenance")
                    groups.setdefault(s, []).append((str(ref), p if isinstance(p, dict) and p else None))
    diffs, unknown = [], []
    for s, items in sorted(groups.items()):
        if not any(p for _, p in items):
            continue
        for ref, p in items:
            if p is None:
                continue
            dirty = _prov_value(p, ("evaluator", "dirty"))
            if isinstance(dirty, list) and dirty:
                diffs.append(f"{ref}: the evaluator differs from its commit ({len(dirty)} file(s), e.g. {dirty[0]}): the card is not used for a verdict")
            elif not isinstance(dirty, list):
                diffs.append(f"{ref}: whether the evaluator differs from its commit was not checked: the card is not used for a verdict")
        for key in PV.COMPARE_KEYS:
            name, known = ".".join(key), {}
            for ref, p in items:
                v = _prov_value(p, key) if p else None
                if v is None or v == PV.UNKNOWN:
                    unknown.append(f"{name} ({ref})")
                else:
                    known.setdefault(json.dumps(v, sort_keys=True), []).append(ref)
            if len(known) > 1:
                diffs.append(f"{'subset ' + s + ': ' if s else ''}the cards differ in {name}: " + "; ".join(f"{json.loads(v)!r} ({', '.join(refs[:3])})" for v, refs in known.items()))
    return list(dict.fromkeys(diffs)), list(dict.fromkeys(unknown))


def judge_clause(clause, exp, cards, root=None):
    """one clause -> dict(ok, effect, diffs, interval, flags, text); subsets are judged one by one"""
    metric = clause["metric"]
    vs = clause.get("vs", exp["control"])
    variant = exp.get("variant") or next(a for a in exp["arms"] if a != exp["control"])
    band = clause.get("band", exp["band"] if metric == exp["metric"] else DEFAULT_BAND)
    subsets = clause.get("subsets") or ([""] if "" in cards[variant] else sorted(cards[variant]))  # a plain clause on subset-only arms: every subset
    parts = []
    for s in subsets:
        vc, cc, how = pair_arms(exp, variant, vs, cards[variant].get(s, []), cards[vs].get(s, []), s)
        effect, diffs, interval = paired_effect(metric, vc, cc, is_lower(metric, exp, root, list(vc) + list(cc)))
        flags = []
        if len(vc) != len(cc):
            flags.append(f"seed counts differ ({len(vc)} vs {len(cc)}): unpaired, counts as one seed")
        if "min_effect" in clause:
            thr = max(clause["min_effect"], band) if metric == exp["metric"] else clause["min_effect"]  # inside the band it is a tie, whatever the clause says
            ok = effect >= thr and (metric != exp["metric"] or effect > band)   # |effect| <= band is a tie, also when the clause asks for no more than the band
            if ok and len(diffs) < 2 and not (exp.get("deterministic") and interval is not None and interval[0] > 0):
                flags.append("one seed")  # a deterministic change (no training seed) is judged by the paired scene-bootstrap interval instead
            if ok and len(diffs) >= 2 and not all(d > 0 for d in diffs):
                flags.append("a seed disagrees")
            if ok and interval is not None and interval[0] <= 0:
                flags.append(f"interval [{interval[0]:+.3f}, {interval[1]:+.3f}] contains 0")
            want = f">= {thr:+.3f}"
        else:
            ok = effect >= -clause["max_loss"]
            want = f">= {-clause['max_loss']:+.3f}"
        parts.append(dict(subset=s, ok=ok, effect=effect, diffs=diffs, interval=interval, flags=flags, how=how, text=f"{metric}{' ' + s if s else ''} {variant}-{vs} {effect:+.4f} (want {want}{', band ' + str(band) if metric == exp['metric'] else ''})"))
    need = clause.get("at_least", len(parts))
    ok = sum(p["ok"] for p in parts) >= need
    flags = sorted({f for p in parts if p["ok"] for f in p["flags"]})
    return dict(metric=metric, vs=vs, kind="min_effect" if "min_effect" in clause else "max_loss", ok=ok, parts=parts, flags=flags, need=need,
                text="; ".join(p["text"] for p in parts) + (f" [{sum(p['ok'] for p in parts)} of {len(parts)} pass, {need} needed]" if len(parts) > 1 else ""))


def decide_experiment(exp, root):
    """verdict from the experiment file and its cards: dict(verdict, effect, why, clauses)"""
    import experiment as X
    cards = {a: X.arm_cards(exp, a, root) for a in exp["arms"]}
    dup = X.duplicate_cards(cards, root, bool(exp.get("deterministic")))
    if dup:
        raise ValueError("a card is listed twice: " + "; ".join(dup))
    bad = incompatible(exp, cards)
    if bad:
        raise ValueError("the arms were not evaluated on the same thing: " + "; ".join(bad) + " (a known difference goes into evaluate.allow)")
    variant = exp.get("variant") or next(a for a in exp["arms"] if a != exp["control"])
    need = {variant, exp["control"]} | {c["vs"] for c in exp["rule"] if c.get("vs") in exp["arms"]}
    none = sorted(a for a in need if not any(cards[a].values()))
    if none:                                                         # the first lane cards of 2026-11 describe their cards in a recipe text: nothing for the tool to read
        raise ValueError(f"arm(s) {', '.join(none)} list no metric cards (`cards` of the arm: paths relative to the problem, '<file>#<name>' for one entry of a file): nothing to judge; "
                         "write them into the card, or decide by hand and say so")
    head_metric, band = exp["metric"], exp["band"]
    def head(arm):  # the whole-split cards, else the subset cards in subset order (then the "seeds" of the headline are the subsets)
        return cards[arm].get("") or [c for s in sorted(cards[arm]) for c in cards[arm][s]]
    if cards[variant].get("") or cards[exp["control"]].get(""):  # whole-split cards: paired seed by seed
        head_v, head_c, how = pair_arms(exp, variant, exp["control"], cards[variant].get("", []), cards[exp["control"]].get("", []))
    else:  # subset-only cards: the subsets are the pairs, both arms must hold the same ones
        if set(cards[variant]) != set(cards[exp["control"]]):
            raise ValueError(f"arm {variant} has the subsets {sorted(cards[variant])}, arm {exp['control']} {sorted(cards[exp['control']])}: nothing to pair")
        head_v, head_c, how = head(variant), head(exp["control"]), "subset"
    effect, diffs, interval = paired_effect(head_metric, head_v, head_c, is_lower(head_metric, exp, root))
    clauses = [judge_clause(c, exp, cards, root) for c in exp["rule"]]
    hows = {how} | {p["how"] for c in clauses for p in c["parts"]}  # the pairing recorded is the weakest used anywhere: a clause against another arm may pair by order
    how = next((h for h in ("unpaired", "position", "seed", "subset") if h in hows), "single")
    guards = [c for c in clauses if c["kind"] == "max_loss" and not c["ok"]]
    hc = next(c for c in clauses if c["metric"] == head_metric and c["kind"] == "min_effect") if any(c["metric"] == head_metric and c["kind"] == "min_effect" for c in clauses) else None
    missed = [c for c in clauses if c["kind"] == "min_effect" and not c["ok"] and c is not hc]
    flags = sorted({f for c in clauses if c["ok"] for f in c["flags"]})
    if hc is None and effect > band and cards[variant].get(""):  # no min_effect clause on the headline (e.g. only a guard): its evidence is judged here
        # (review 2026-10-31, T1); only for whole-split cards: with subset-only cards the "seeds" of head() are subsets, which a guard's at_least judges
        if len(diffs) < 2 and not (exp.get("deterministic") and interval is not None and interval[0] > 0):
            flags.append("one seed")
        if len(diffs) >= 2 and not all(d > 0 for d in diffs):
            flags.append("a seed disagrees")
        if interval is not None and interval[0] <= 0:
            flags.append(f"interval [{interval[0]:+.3f}, {interval[1]:+.3f}] contains 0")
        flags = sorted(set(flags))
    if guards:
        verdict, why = "CUT", "guard violated: " + "; ".join(c["text"] for c in guards)
    elif hc is not None and not hc["ok"]:
        if len(hc["parts"]) > 1 and all(abs(p["effect"]) <= band for p in hc["parts"]):
            verdict, why = "TIE", f"headline {head_metric}: every subset inside the band {band}: take the cheaper arm"
        elif len(hc["parts"]) > 1:
            verdict, why = "CUT", "rule not met: " + hc["text"]
        elif abs(effect) <= band:
            verdict, why = "TIE", f"headline {head_metric} {variant}-{exp['control']} {effect:+.4f} inside the band {band}: take the cheaper arm"
        elif effect < -band:
            verdict, why = "CUT", f"headline {head_metric} {variant}-{exp['control']} {effect:+.4f} beyond the band {band} in the worse direction"
        else:
            verdict, why = "CUT", "rule not met: " + hc["text"]
    elif hc is None and abs(effect) <= band and not missed:
        verdict, why = "TIE", f"headline {head_metric} {variant}-{exp['control']} {effect:+.4f} inside the band {band}: take the cheaper arm"
    elif hc is None and effect < -band:
        verdict, why = "CUT", f"headline {head_metric} {variant}-{exp['control']} {effect:+.4f} beyond the band {band} in the worse direction"
    elif missed:
        pending = [c for c in missed if any(p["flags"] for p in c["parts"])]
        verdict, why = ("INCONCLUSIVE" if pending else "CUT"), "rule not met: " + "; ".join(c["text"] for c in missed)
    elif flags:
        verdict, why = "INCONCLUSIVE", f"headline {head_metric} {effect:+.4f} beyond the band but " + ", ".join(flags) + ": add a seed, do not adopt"
    elif hc is not None and len(hc["parts"]) > 1:
        verdict, why = "KEEP", f"headline {head_metric} by subsets ({sum(p['ok'] for p in hc['parts'])} of {len(hc['parts'])} pass, {hc['need']} needed): " + "; ".join(f"{p['subset']} {p['effect']:+.4f}" for p in hc["parts"]) + "; every clause holds"
    else:
        verdict, why = "KEEP", f"headline {head_metric} {variant}-{exp['control']} {effect:+.4f} beyond the band {band}" + (f", interval [{interval[0]:+.3f}, {interval[1]:+.3f}]" if interval else "") + f", {len(diffs)} paired seed(s); every clause holds"
    pred = exp.get("prediction", {})
    pred_txt = ""
    if isinstance(pred.get("value"), (int, float)) and pred.get("vs") == exp["control"]:
        pred_txt = f"prediction {pred['value']:+.3f}, measured {effect:+.4f} (error {effect - pred['value']:+.4f})"
    subset_effects = {p["subset"]: round(p["effect"], 5) for p in hc["parts"]} if hc is not None and len(hc["parts"]) > 1 else None
    waived, unknown = provenance_check(exp, cards)               # differences exist here only when evaluate.allow named "provenance" (else the decision was refused above)
    if waived:
        why += "; provenance differences waived by evaluate.allow: " + " | ".join(waived)[:400]
    if unknown:                                                  # values the provenance blocks of the cards do not tell (never a refusal)
        why += "; provenance: cannot tell " + ", ".join(sorted({k.split(" ")[0] for k in unknown}))
    return dict(id=exp["id"], cannot_tell=unknown, provenance_waived=waived, verdict=verdict, metric=head_metric, variant=variant, control=exp["control"], effect=round(effect, 5), diffs=[round(d, 5) for d in diffs], pairing=how,
                subset_effects=subset_effects, effect_note="effect is pooled over the subset cards; the verdict uses subset_effects" if subset_effects else "",
                interval=[round(x, 5) for x in interval] if interval else None, why=why, prediction=pred_txt, clauses=[dict(text=c["text"], ok=c["ok"], flags=c["flags"]) for c in clauses],
                date=time.strftime("%Y-%m-%d %H:%M"))


# ------------------------------------------------------------------ the gate: a verdict needs finished runs and a rule that was fixed before them


def listed_runs(exp):
    """the run ids the variant and the control arm list"""
    variant = exp.get("variant") or next(a for a in exp["arms"] if a != exp["control"])
    return [r for a in (variant, exp["control"]) for r in (exp["arms"][a].get("run_ids") or [])]


def status_problem(rid, status):
    """why the run rid cannot carry a verdict yet ('' when its ledger row has status ok); status: {run id: ledger status}"""
    if status.get(rid) == "ok":
        return ""
    s = status.get(rid)
    return f"run {rid} has status {'missing' if s is None else s or 'empty'}: nothing to decide yet" + (f" (no ledger row {rid})" if s is None else "")


def run_gate(exp, rows, records):
    """(refusals, registered): the runs behind a verdict. Each run id of the variant and the control arm needs a ledger row with status ok. A run record may carry the
    `card_sha` of the rule at launch (run.py): it must equal experiment.rule_sha(card) now. registered: at least one listed run carries a card_sha"""
    import experiment as X
    import run as RUN
    now, status, bad, registered = X.rule_sha(exp), {r["id"]: r.get("status") or "" for r in rows}, [], False
    for rid in listed_runs(exp):
        if why := status_problem(rid, status):
            bad.append(why)
            continue
        rec = RUN.record(os.path.join(records, rid))
        sha = rec.get("card_sha")
        if isinstance(sha, str) and sha and rec.get("card_id") in (None, "", exp.get("id")):    # a run that several cards list (E266.b0 in E266 and E267) is registered for the card that was found first
            registered = True
            if sha != now:
                bad.append(f"the rule changed after run {rid} was launched (card_sha {sha}, the card now {now})")
    return bad, registered


def gated_decision(exp, root, rows, records, by_hand=""):
    """decide_experiment behind the gate; ValueError with the reasons when a listed run is not ok or the rule changed after its launch (by_hand bypasses both). The decision
    carries rule_sha, registered (a run record held the card_sha of this rule) and by_hand (the reason)"""
    import experiment as X
    bad, registered = ([], False) if by_hand else run_gate(exp, rows, records)
    if bad:
        raise ValueError("; ".join(bad))
    d = decide_experiment(exp, root)
    d.update(rule_sha=X.rule_sha(exp), registered=registered, by_hand=by_hand)
    if not by_hand and not registered:
        d["why"] += "; rule not registered at launch (no card_sha in the run records)"
    return d


def decision_path(card_path):
    """decision.json next to the card file (experiment.md -> decision.json; experiment_x.json -> decision_x.json)"""
    return os.path.join(os.path.dirname(card_path), "decision.json" if card_path.endswith(".md") else os.path.basename(card_path).replace("experiment", "decision", 1))


def previous_verdicts(out):
    """the verdicts an earlier decision.json at `out` holds (its own `previous` list, then itself), the newest last, at most 10: a new decision does not erase the old one"""
    try:
        with open(out, encoding="utf-8") as f:
            old = json.load(f)
    except FileNotFoundError:
        return []
    except (OSError, ValueError) as e:
        print(f"  warning: the earlier {os.path.basename(out)} cannot be read ({e}): its verdict is not kept")
        return []
    if not isinstance(old, dict) or not old.get("verdict"):
        return []
    prev = [p for p in (old["previous"] if isinstance(old.get("previous"), list) else []) if isinstance(p, dict)]
    prev.append({k: old.get(k) for k in ("verdict", "effect", "date", "rule_sha", "by_hand")})
    return prev[-10:]


def write_records(path, exp, d, root):
    """the records of a verdict, in this order: the decision cell of the variant arm's ledger rows (a failure stops here, exit 1: no record that the ledger does not know),
    decision.json (atomic; the earlier verdicts in `previous`), then result.record(): results.md, the schedule link, the result line of the cited research notes (side records:
    a failure is a printed warning). Returns the exit code"""
    import ledger as L
    import project as P
    here = os.path.dirname(os.path.abspath(__file__))
    out, folder = decision_path(path), os.path.dirname(path)
    token = decision_token(d["verdict"], d["date"][:10], d["effect"], [f"by hand: {d['by_hand']}"] if d.get("by_hand") else [])
    old = {r["id"]: r.get("decision", "") for r in L.load()}
    for rid in exp["arms"][d["variant"]].get("run_ids", []):
        if rid not in old:  # ledger.py note would create an empty 'planned' row for an id that never ran
            print(f"  warning: no ledger row {rid}: the verdict is only in {out} (fix run_ids of the variant arm in the experiment file)")
            continue
        r = subprocess.run([sys.executable, os.path.join(here, "ledger.py"), "note", rid, "--decision", ledger_decision(old[rid], token)])
        if r.returncode:
            print(f"error: the ledger note of {rid} failed (exit {r.returncode}): nothing else written, run decide.py again")
            return 1
    d["previous"] = previous_verdicts(out)
    try:
        P.atomic_write(out, json.dumps(d, indent=1) + "\n")
    except OSError as e:
        print(f"error: {out} not written ({e})")
        return 1
    print("  written:", out)
    import result as R
    R.record(path, exp, d, root)                          # results.md, the schedule link, the research notes: side records, a failure there is a printed warning
    return 0


def decide_card(a):
    """--experiment: check the card, the gate, the rule; --write records the verdict. Returns the exit code"""
    import experiment as X
    import ledger as L
    import project as P
    exp, path = X.load(a.experiment)
    bad = X.check(exp, path)
    if bad:
        sys.exit(f"{exp.get('id')}: experiment file not ready: " + "; ".join(bad))
    root, by_hand = X.problem_of(path), " ".join((a.by_hand or "").split())
    try:
        d = gated_decision(exp, root, L.load(), P.path("run_records"), by_hand)
    except ValueError as e:  # a run that is not ok, a rule that changed after its launch, cards that disagree about a metric's direction
        sys.exit(f"{exp.get('id')}: not decided: {e}")
    print(f"{d['id']}: {d['verdict']}: {d['why']}")
    for c in d["clauses"]:
        print(f"  {'ok  ' if c['ok'] else 'MISS'} {c['text']}" + (f"  flags: {', '.join(c['flags'])}" if c["flags"] else ""))
    if d["prediction"]:
        print("  " + d["prediction"])
    if by_hand:
        print(f"  by hand: {by_hand} (the gate was not applied)")
    return write_records(path, exp, d, root) if a.write else 0


def decide_typed(a, ap):
    """--variant / --control: typed scores; --write puts the token on the ledger row of --id, which must exist with status ok. Returns the exit code"""
    if not (a.variant and a.control):
        ap.error("--experiment, or --variant and --control")
    v = [float(x) for x in a.variant.split(",")]
    c = [float(x) for x in a.control.split(",")]
    verdict, effect, why = decide(v, c, a.band, a.lower_better, a.cost_variant, a.cost_control)
    print(f"{verdict}: effect {effect:+.4f} ({why})")
    if a.write and a.id:
        import ledger as L
        rows = L.load()
        problem = status_problem(a.id, {r["id"]: r.get("status") or "" for r in rows})  # ledger.py note would create an empty 'planned' row for an id that never ran
        if problem:
            print(f"refused: {problem}: the verdict is only printed")
            return 1
        band = f"{a.band:.3f}" if round(a.band, 3) == a.band else f"{a.band:g}"
        token = decision_token(verdict, time.strftime("%Y-%m-%d"), effect, ["typed scores", f"band {band}", "by hand: typed scores"])
        old = next(r.get("decision", "") for r in rows if r["id"] == a.id)
        subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "ledger.py"), "note", a.id, "--decision", ledger_decision(old, token)], check=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", help="experiment id (E240), folder, experiment.md or experiment.json")
    ap.add_argument("--variant", help="comma list, one score per seed")
    ap.add_argument("--control")
    ap.add_argument("--band", type=float, default=0.010)
    ap.add_argument("--lower-better", action="store_true")
    ap.add_argument("--cost-variant", type=float)
    ap.add_argument("--cost-control", type=float)
    ap.add_argument("--id", default="")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--by-hand", default=None, metavar="REASON", help="--experiment: decide without the gate (finished runs, rule fixed before them) and label the verdict 'by hand: REASON'")
    a = ap.parse_args()
    if a.by_hand is not None and (not a.experiment or not a.by_hand.strip()):
        ap.error("--by-hand REASON belongs to --experiment and needs a reason (a typed verdict is always labelled by hand)")
    return decide_card(a) if a.experiment else decide_typed(a, ap)


if __name__ == "__main__":
    sys.exit(main())
