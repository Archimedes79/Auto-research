"""Decision rule as code: KEEP / CUT / TIE / INCONCLUSIVE, from an experiment file and its metric cards, or from typed per-seed scores.

    python decide.py --experiment E240 [--write]                     the rule clauses of experiments/E240_*/experiment.json applied to the cards
    python decide.py --variant 0.746,0.741 --control 0.725,0.722 [--band 0.010] [--lower-better] [--cost-variant 31 --cost-control 628] [--id E123.D7 --write]

Rule (autoresearch/METHOD.md, P3 step 5; schema autoresearch/schemas/experiment.schema.json):
  * effect = mean(variant) - mean(control) on the headline metric; with equal seed counts the seeds are paired (seed i against seed i)
  * |effect| <= band                                   -> TIE: take the cheaper arm
  * beyond the band in the worse direction              -> CUT
  * beyond the band in the better direction, but one seed only, a paired seed on the other side, or the paired scene-bootstrap interval
    (compare block of the card, when the card was made with evaluate.py --compare <control>) contains 0 -> INCONCLUSIVE: add a seed, do not adopt
  * beyond the band in the better direction with agreeing seeds (and interval) -> KEEP, if every clause of the rule holds:
      min_effect clause: variant - vs >= min_effect (with agreeing seeds); max_loss clause (a guard): variant - vs >= -max_loss;
      a clause with subsets passes when at_least (default: all) of its subsets pass. A violated guard -> CUT; a missed min_effect -> CUT ("rule not met")
  * deterministic: true (the arms share the checkpoint, e.g. a post-processing change): no training seed exists, so a single pair with an
    interval that excludes 0 counts as agreeing seeds
  * bands: the experiment's band for its headline metric, a clause's own band, else 0.010; lower-is-better metrics are known by name (LOWER)
  * --write writes decision.json next to the experiment file and appends "decide.py <date>: VERDICT (effect)" to the decision column of the
    variant arm's ledger rows (run_ids); a prose decision already there is kept in front of it
This is the pre-registered decision, not a significance test: intervals come from the problem's evaluation (paired scene bootstrap).
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LOWER = {"vert100", "false_len_frac", "dup_frac", "ms", "ms_per_frame", "asm_ms", "map_err", "path_err", "ece"}
DEFAULT_BAND = 0.010


def decide(variant, control, band=0.010, lower_better=False, cost_variant=None, cost_control=None):
    """typed scores, one per seed (the loop-18 interface)"""
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


def is_lower(metric, exp=None):
    """lower is better: the experiment's direction for its headline metric, else the known names (LOWER)"""
    if exp is not None and metric == exp.get("metric"):
        return exp.get("direction") == "lower"
    return metric in LOWER


def paired_effect(metric, v_cards, c_cards, lower=None):
    """(effect, paired diffs, interval or None) of a metric between two lists of (metrics, card, ref); the sign is 'higher is better'"""
    sgn = -1.0 if (metric in LOWER if lower is None else lower) else 1.0
    v = [m[metric] for m, _, _ in v_cards]
    c = [m[metric] for m, _, _ in c_cards]
    if not v or not c:
        raise KeyError(f"no card for {metric}")
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


def judge_clause(clause, exp, cards):
    """one clause -> dict(ok, effect, diffs, interval, flags, text); subsets are judged one by one"""
    import experiment as X  # noqa: F401  (same folder)
    metric = clause["metric"]
    vs = clause.get("vs", exp["control"])
    variant = exp.get("variant") or next(a for a in exp["arms"] if a != exp["control"])
    band = clause.get("band", exp["band"] if metric == exp["metric"] else DEFAULT_BAND)
    subsets = clause.get("subsets") or ([""] if "" in cards[variant] else sorted(cards[variant]))  # a plain clause on subset-only arms: every subset
    parts = []
    for s in subsets:
        vc, cc = cards[variant].get(s, []), cards[vs].get(s, [])
        effect, diffs, interval = paired_effect(metric, vc, cc, is_lower(metric, exp))
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
    variant = exp.get("variant") or next(a for a in exp["arms"] if a != exp["control"])
    head_metric, band = exp["metric"], exp["band"]
    sgn = 1.0  # paired_effect already turns lower-is-better around; exp["direction"] is for the reader
    def head(arm):  # the whole-split cards, else the subset cards in subset order (then the "seeds" of the headline are the subsets)
        return cards[arm].get("") or [c for s in sorted(cards[arm]) for c in cards[arm][s]]
    effect, diffs, interval = paired_effect(head_metric, head(variant), head(exp["control"]), is_lower(head_metric, exp))
    effect *= sgn
    clauses = [judge_clause(c, exp, cards) for c in exp["rule"]]
    guards = [c for c in clauses if c["kind"] == "max_loss" and not c["ok"]]
    hc = next(c for c in clauses if c["metric"] == head_metric and c["kind"] == "min_effect") if any(c["metric"] == head_metric and c["kind"] == "min_effect" for c in clauses) else None
    missed = [c for c in clauses if c["kind"] == "min_effect" and not c["ok"] and c is not hc]
    flags = sorted({f for c in clauses if c["ok"] for f in c["flags"]})
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
    return dict(id=exp["id"], verdict=verdict, metric=head_metric, variant=variant, control=exp["control"], effect=round(effect, 5), diffs=[round(d, 5) for d in diffs],
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
        d = decide_experiment(exp, root)
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
