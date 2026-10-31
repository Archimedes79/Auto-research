"""Decision rule as code: KEEP / CUT / TIE / INCONCLUSIVE, from an experiment file and its metric cards, or from typed per-seed scores.

    python decide.py --experiment E240 [--write]                     the rule clauses of experiments/E240_*/experiment.json applied to the cards
    python decide.py --variant 0.746,0.741 --control 0.725,0.722 [--band 0.010] [--lower-better] [--cost-variant 31 --cost-control 628] [--id E123.D7 --write]

Rule (autoresearch/METHOD.md, P3 step 5; schema autoresearch/schemas/experiment.schema.json):
  * effect = mean(variant) - mean(control) on the headline metric; with equal seed counts the seeds are paired BY SEED IDENTITY (pair_arms: the
    card's `seed`, else the arm's / experiment's `seeds` list; a seed only one arm ran, a duplicate, or two or more seeds without any identity are
    refused unless the experiment declares "pairing": "position"); the typed interface pairs by position. Values must be finite numbers
  * the cards of all arms must be measurements of the same thing (`incompatible`: split = evaluate.split, subset, denominator, protocol;
    evaluate.allow lists a known difference), else nothing is decided
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
  * --write writes decision.json next to the experiment file and appends "decide.py <date>: VERDICT (effect)" to the decision column of the
    variant arm's ledger rows (run_ids); a prose decision already there is kept in front of it
This is the pre-registered decision, not a significance test: intervals come from the problem's evaluation (paired scene bootstrap).
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LOWER = {"ms", "ms_per_frame"}  # generic cost names; problem metrics are listed in project.json "lower_is_better"
DEFAULT_BAND = 0.010


def decide(variant, control, band=0.010, lower_better=False, cost_variant=None, cost_control=None):
    """typed scores, one per seed (the loop-18 interface): the position IS the pairing, seed i of the variant against seed i of the control
    (experiment files pair by seed identity instead: pair_arms)"""
    if not variant or not control or not all(math.isfinite(x) for x in [*variant, *control, band]):
        raise ValueError("scores and band must be finite numbers, at least one score per arm")
    sgn = -1.0 if lower_better else 1.0
    mv, mc = sum(variant) / len(variant), sum(control) / len(control)
    effect = sgn * (mv - mc)
    if len(variant) == len(control):
        diffs = [sgn * (v - c) for v, c in zip(variant, control)]
    else:
        diffs = [effect]
    if abs(effect) <= band:
        if cost_variant is not None and cost_control is not None:
            cheaper = "variant" if cost_variant < cost_control else "control"
            return "TIE", effect, f"inside the noise band {band}: take the cheaper one ({cheaper}: {cost_variant} vs {cost_control})"
        return "TIE", effect, f"inside the noise band {band}: take the simpler / cheaper one"
    same = all((d > 0) == (effect > 0) and abs(d) > 0 for d in diffs)
    if len(diffs) < 2 or not same:
        return "INCONCLUSIVE", effect, "beyond the band but " + ("only one seed" if len(diffs) < 2 else "a seed disagrees") + ": add a seed, do not adopt yet"
    return ("KEEP" if effect > 0 else "CUT"), effect, f"beyond the band {band} on all {len(diffs)} paired seeds"


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
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in [*v, *c]):
        raise ValueError(f"{metric}: a card holds a value that is not a finite number (NaN, infinity or text)")
    effect = sgn * (sum(v) / len(v) - sum(c) / len(c))
    diffs = [sgn * (a - b) for a, b in zip(v, c)] if len(v) == len(c) else [effect]
    interval = None
    for m, card, _ in v_cards:  # the paired scene bootstrap of evaluate.py --compare, when the card compared against this control
        cmp = (card or {}).get("compare") or {}
        pr = cmp.get("paired", {}).get(metric)
        c_names = {(cc or {}).get("name") for _, cc, _ in c_cards}
        if isinstance(pr, dict) and "lo" in pr and cmp.get("name") in c_names:
            lo, hi = sgn * pr["lo"], sgn * pr["hi"]
            interval = (min(lo, hi), max(lo, hi))
            break
    return effect, diffs, interval


def seed_ids(exp, arm, cards):
    """the seed identity of each card of one arm, as strings in card order, or None when unknown: the card's own `seed`, else the arm's
    `seeds`, else the experiment's `seeds`, when that list is as long as the cards"""
    own = [(c or {}).get("seed") for _, c, _ in cards]
    if cards and all(s is not None for s in own):
        return [str(s) for s in own]
    for declared in (arm.get("seeds"), exp.get("seeds")):
        if isinstance(declared, list) and len(declared) == len(cards) and cards:
            return [str(s) for s in declared]
    return None


def pair_arms(exp, a, b, cards_a, cards_b):
    """(cards_a, cards_b in the order of cards_a, how): the cards of arms a and b paired SEED BY SEED. Equal counts are paired by seed
    identity (seed_ids; duplicates and seeds that only one arm ran are refused: ValueError). Without identities on both sides two or more
    seeds are refused too, unless the experiment declares "pairing": "position" (card i against card i: then the order of the lists is
    the pairing, and the decision says so). Unequal counts are not paired (how = "unpaired": the caller counts them as one seed)."""
    if len(cards_a) != len(cards_b):
        return cards_a, cards_b, "unpaired"
    ids_a, ids_b = seed_ids(exp, exp["arms"][a], cards_a), seed_ids(exp, exp["arms"][b], cards_b)
    if ids_a is not None and ids_b is not None:
        for arm, ids in ((a, ids_a), (b, ids_b)):
            if len(set(ids)) != len(ids):
                raise ValueError(f"arm {arm} lists a seed twice ({', '.join(ids)}): the arms cannot be paired")
        if set(ids_a) != set(ids_b):
            raise ValueError(f"the seeds of arm {a} ({', '.join(ids_a)}) and arm {b} ({', '.join(ids_b)}) are not the same: a seed that only one arm ran has no pair")
        by_seed = dict(zip(ids_b, cards_b))
        return cards_a, [by_seed[s] for s in ids_a], "seed" if len(cards_a) > 1 else "single"
    if len(cards_a) < 2:
        return cards_a, cards_b, "single"
    if exp.get("pairing") == "position":
        return cards_a, cards_b, "position"
    raise ValueError(f"arms {a} and {b} have {len(cards_a)} cards each but no seed identities to pair them by: give every card a `seed`, or the arms (or the "
                     "experiment) a `seeds` list as long as their cards, or declare \"pairing\": \"position\" in the experiment file (card i against card i)")


def incompatible(exp, cards):
    """the reasons why the cards of the arms are not measurements of the same thing, from the cards' own fields: split (must be the experiment's
    evaluate.split), subset (of whole-split cards, when the card has the field: evaluate.subset), the denominator (frames / n) and the optional
    protocol (what the evaluator says about ground truth, cache, frame list) must be the same in every arm for one subset key. A bare metrics dict
    holds no fields and passes. exp["evaluate"]["allow"] lists fields that are knowingly different (split | subset | denominator | protocol)."""
    ev = exp.get("evaluate", {})
    allow = set(ev.get("allow", []))
    out, seen = [], {}
    for by_key in cards.values():
        for s, lst in by_key.items():
            for _, card, ref in lst:
                if not card:
                    continue
                if card.get("split") is not None and "split" not in allow and card["split"] != ev.get("split"):
                    out.append(f"{ref}: split {card['split']!r}, the experiment evaluates {ev.get('split')!r}")
                if s == "" and "subset" in card and "subset" not in allow and (card["subset"] or "") != ev.get("subset", ""):
                    out.append(f"{ref}: subset {card['subset']!r}, the experiment evaluates {ev.get('subset', '')!r}")
                for field, value in (("denominator", card.get("frames", card.get("n"))), ("protocol", card.get("protocol"))):
                    if value is not None and field not in allow:
                        seen.setdefault((s, field), {}).setdefault(json.dumps(value, sort_keys=True), []).append(str(ref))
    for (s, field), values in sorted(seen.items()):
        if len(values) > 1:
            out.append(f"{'subset ' + s + ': ' if s else ''}the cards differ in {field}: " + "; ".join(f"{json.loads(v)!r} ({', '.join(refs[:2])})" for v, refs in values.items()))
    return out


def judge_clause(clause, exp, cards, root=None):
    """one clause -> dict(ok, effect, diffs, interval, flags, text); subsets are judged one by one"""
    import experiment as X  # noqa: F401  (same folder)
    metric = clause["metric"]
    vs = clause.get("vs", exp["control"])
    variant = exp.get("variant") or next(a for a in exp["arms"] if a != exp["control"])
    band = clause.get("band", exp["band"] if metric == exp["metric"] else DEFAULT_BAND)
    subsets = clause.get("subsets") or ([""] if "" in cards[variant] else sorted(cards[variant]))  # a plain clause on subset-only arms: every subset
    parts = []
    for s in subsets:
        vc, cc, _ = pair_arms(exp, variant, vs, cards[variant].get(s, []), cards[vs].get(s, []))
        effect, diffs, interval = paired_effect(metric, vc, cc, is_lower(metric, exp, root, list(vc) + list(cc)))
        flags = []
        if len(vc) != len(cc):
            flags.append(f"seed counts differ ({len(vc)} vs {len(cc)}): unpaired, counts as one seed")
        if "min_effect" in clause:
            thr = max(clause["min_effect"], band) if metric == exp["metric"] else clause["min_effect"]  # inside the band it is a tie, whatever the clause says
            ok = effect >= thr
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
        parts.append(dict(subset=s, ok=ok, effect=effect, diffs=diffs, interval=interval, flags=flags, text=f"{metric}{' ' + s if s else ''} {variant}-{vs} {effect:+.4f} (want {want}{', band ' + str(band) if metric == exp['metric'] else ''})"))
    need = clause.get("at_least", len(parts))
    ok = sum(p["ok"] for p in parts) >= need
    flags = sorted({f for p in parts if p["ok"] for f in p["flags"]})
    return dict(metric=metric, vs=vs, kind="min_effect" if "min_effect" in clause else "max_loss", ok=ok, parts=parts, flags=flags, need=need,
                text="; ".join(p["text"] for p in parts) + (f" [{sum(p['ok'] for p in parts)} of {len(parts)} pass, {need} needed]" if len(parts) > 1 else ""))


def decide_experiment(exp, root):
    """verdict from the experiment file and its cards: dict(verdict, effect, why, clauses)"""
    import experiment as X
    cards = {a: X.arm_cards(exp, a, root) for a in exp["arms"]}
    bad = incompatible(exp, cards)
    if bad:
        raise ValueError("the arms were not evaluated on the same thing: " + "; ".join(bad) + " (a known difference goes into evaluate.allow)")
    variant = exp.get("variant") or next(a for a in exp["arms"] if a != exp["control"])
    head_metric, band = exp["metric"], exp["band"]
    sgn = 1.0  # paired_effect already turns lower-is-better around; exp["direction"] is for the reader
    def head(arm):  # the whole-split cards, else the subset cards in subset order (then the "seeds" of the headline are the subsets)
        return cards[arm].get("") or [c for s in sorted(cards[arm]) for c in cards[arm][s]]
    if cards[variant].get("") or cards[exp["control"]].get(""):  # whole-split cards: paired seed by seed
        head_v, head_c, how = pair_arms(exp, variant, exp["control"], cards[variant].get("", []), cards[exp["control"]].get("", []))
    else:  # subset-only cards: the subsets are the pairs, both arms must hold the same ones
        if set(cards[variant]) != set(cards[exp["control"]]):
            raise ValueError(f"arm {variant} has the subsets {sorted(cards[variant])}, arm {exp['control']} {sorted(cards[exp['control']])}: nothing to pair")
        head_v, head_c, how = head(variant), head(exp["control"]), "subset"
    effect, diffs, interval = paired_effect(head_metric, head_v, head_c, is_lower(head_metric, exp, root))
    effect *= sgn
    clauses = [judge_clause(c, exp, cards, root) for c in exp["rule"]]
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
    return dict(id=exp["id"], verdict=verdict, metric=head_metric, variant=variant, control=exp["control"], effect=round(effect, 5), diffs=[round(d, 5) for d in diffs], pairing=how,
                subset_effects=subset_effects, effect_note="effect is pooled over the subset cards; the verdict uses subset_effects" if subset_effects else "",
                interval=[round(x, 5) for x in interval] if interval else None, why=why, prediction=pred_txt, clauses=[dict(text=c["text"], ok=c["ok"], flags=c["flags"]) for c in clauses],
                date=time.strftime("%Y-%m-%d %H:%M"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", help="experiment id (E240), folder or experiment.json")
    ap.add_argument("--variant", help="comma list, one score per seed")
    ap.add_argument("--control")
    ap.add_argument("--band", type=float, default=0.010)
    ap.add_argument("--lower-better", action="store_true")
    ap.add_argument("--cost-variant", type=float)
    ap.add_argument("--cost-control", type=float)
    ap.add_argument("--id", default="")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    if a.experiment:
        import experiment as X
        exp, path = X.load(a.experiment)
        bad = X.check(exp, path)
        if bad:
            sys.exit(f"{exp.get('id')}: experiment file not ready: " + "; ".join(bad))
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(path))))
        try:
            d = decide_experiment(exp, root)
        except ValueError as e:  # e.g. cards that disagree about a metric's direction
            sys.exit(f"{exp.get('id')}: not decided: {e}")
        print(f"{d['id']}: {d['verdict']}: {d['why']}")
        for c in d["clauses"]:
            print(f"  {'ok  ' if c['ok'] else 'MISS'} {c['text']}" + (f"  flags: {', '.join(c['flags'])}" if c["flags"] else ""))
        if d["prediction"]:
            print("  " + d["prediction"])
        if a.write:
            out = os.path.join(os.path.dirname(path), os.path.basename(path).replace("experiment", "decision", 1))  # experiment_x.json -> decision_x.json
            json.dump(d, open(out, "w", encoding="utf-8"), indent=1)
            print("  written:", out)
            import ledger as L
            old = {r["id"]: r.get("decision", "") for r in L.load()}
            for rid in exp["arms"][d["variant"]].get("run_ids", []):
                if rid not in old:  # ledger.py note would create an empty 'planned' row for an id that never ran
                    print(f"  warning: no ledger row {rid}: the verdict is only in {out} (fix run_ids of the variant arm in the experiment file)")
                    continue
                line = f"decide.py {d['date'][:10]}: {d['verdict']} (effect {d['effect']:+.4f})"
                prose = old.get(rid, "").split(" | decide.py")[0].strip()  # a written prose decision is kept, an earlier decide.py verdict is replaced
                text = (prose + " | " + line) if prose and not prose.startswith("decide.py") else line
                subprocess.run([sys.executable, os.path.join(here, "ledger.py"), "note", rid, "--decision", text[:400]], check=True)
        return 0
    if not (a.variant and a.control):
        ap.error("--experiment, or --variant and --control")
    v = [float(x) for x in a.variant.split(",")]
    c = [float(x) for x in a.control.split(",")]
    verdict, effect, why = decide(v, c, a.band, a.lower_better, a.cost_variant, a.cost_control)
    line = f"{verdict}: effect {effect:+.4f} ({why})"
    print(line)
    if a.write and a.id:
        import ledger as L
        if a.id not in {r["id"] for r in L.load()}:  # ledger.py note would create an empty 'planned' row for an id that never ran
            print(f"warning: no ledger row {a.id}: the verdict is only printed")
            return 0
        subprocess.run([sys.executable, os.path.join(here, "ledger.py"), "note", a.id, "--decision", line], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
