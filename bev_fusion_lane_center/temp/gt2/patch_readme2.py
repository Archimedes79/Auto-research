p = "evaluation/README.md"
lines = open(p, encoding="utf-8").read().split("\n")
for k, l in enumerate(lines):
    if l.startswith("| `qends_eval.py`") and "--snap" not in l:
        lines[k] = l.rstrip(" |") + "; `score --snap` (E311): connectors drawn from the end of the nearest delivered piece wired to the entry to the start of the piece wired to the exit, one piece per end (refinements / controls `--snap-all --snap-guard --snap-dedup --snap-nogeom --snap-both`); its card goes to `ends_card_snap_*.json`, the card of record `ends_card.json` is only written by the unsnapped read-out at p >= 0.5 |"
    if l.startswith("| `jends_eval.py`") and "--snap" not in l:
        lines[k] = l.rstrip(" |") + "; `score --snap ...` as `qends_eval.py` |"
open(p, "w", encoding="utf-8").write("\n".join(lines))
print("ok")
