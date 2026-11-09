"""Scratch (critic, 2026-11-07): plant bugs in a COPY of losses.py and run the copy's test. Never touches the real files.
Run: wsl bash -c "cd /mnt/c/Auto-research/bev_fusion/temp/critic_line_bce && PYTHONPATH=/mnt/c/Auto-research ~/miniconda3/envs/gcar/bin/python mutate.py [test_file] [mutant prefixes]"
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REAL = os.path.join(HERE, "..", "..", "training", "losses.py")
COPY = os.path.join(HERE, "training", "losses.py")
TEST = sys.argv[1] if len(sys.argv) > 1 else "tests/test_line_bce.py"

DIST = "neg_m = ((target <= 0.5) & (dist >= ignore)).to(logits.dtype)"
POS = "pos_m = (target > 0.5).to(logits.dtype)"
WW = "ww = w * (pos * pos_m + neg_m)"
RET = 'return _wsum(F.binary_cross_entropy_with_logits(logits, target, reduction="none"), ww) / ww.sum((0, 2, 3)).clamp(min=EPS)'
CALL = 'lb = line_bce_loss(ch("line"), tg("line"), tg("dist"), w, cfg.line_bce_pos, cfg.line_bce_ignore)'
GRP = 'groups["lines"] = groups["lines"] + cfg.line_bce * lb.mean()'
IF = "if cfg.line_bce > 0.0:"
G0 = 'groups["lines"] = dice.mean() + hub.mean()'
NAMES = 'for j, n in enumerate(names["line"]):\n                terms[f"line_bce.{n}"] = lb[j]'
M = [
    ("M01 >= -> > in the ignore test", DIST, DIST.replace(">= ignore", "> ignore")),
    ("M02 positive threshold >0.5 -> >0.0 (equivalent on binary GT)", POS, POS.replace("> 0.5", "> 0.0")),
    ("M02b positive threshold >0.5 -> >0.9", POS, POS.replace("> 0.5", "> 0.9")),
    ("M03 positive weight on the negatives", WW, "ww = w * (pos_m + pos * neg_m)"),
    ("M04 normalise by the plain cell count", RET, RET.replace("ww.sum((0, 2, 3)).clamp(min=EPS)", "_wcount(w, logits)")),
    ("M05 drop the map weight w (numerator and denominator)", WW, "ww = (pos * pos_m + neg_m)"),
    ("M05b map weight w missing in the denominator only", RET, RET.replace("ww.sum((0, 2, 3))", "(pos * pos_m + neg_m).sum((0, 2, 3))")),
    ("M05c map weight w missing in the numerator only", RET, RET.replace('reduction="none"), ww)', 'reduction="none"), (pos * pos_m + neg_m))')),
    ("M06 wrong distance channel (reversed order)", CALL, CALL.replace('tg("dist")', 'tg("dist").flip(1)')),
    ("M07 wrong sign of the term", GRP, GRP.replace("+ cfg.line_bce", "- cfg.line_bce")),
    ("M08 term without its weight", GRP, GRP.replace("cfg.line_bce * lb.mean()", "lb.mean()")),
    ("M09 forget the ignore band (all non-line cells negative)", DIST, "neg_m = (target <= 0.5).to(logits.dtype)"),
    ("M10 also applied with line_bce = 0 (>= 0.0)", IF, "if cfg.line_bce >= 0.0:"),
    ("M11 default line_bce 0.0 -> 1.0", "line_bce: float = 0.0 ", "line_bce: float = 1.0 "),
    ("M12 default line_bce_ignore 1.0 -> 0.5", "line_bce_ignore: float = 1.0 ", "line_bce_ignore: float = 0.5 "),
    ("M12b default line_bce_pos 3.0 -> 1.0", "line_bce_pos: float = 3.0 ", "line_bce_pos: float = 1.0 "),
    ("M13 pos and ignore swapped in map_terms", CALL, CALL.replace("cfg.line_bce_pos, cfg.line_bce_ignore", "cfg.line_bce_ignore, cfg.line_bce_pos")),
    ("M14 sigmoid applied before the BCE-with-logits", CALL, CALL.replace('ch("line")', 'torch.sigmoid(ch("line"))')),
    ("M15 lb.mean() -> lb.sum()", GRP, GRP.replace("lb.mean()", "lb.sum()")),
    ("M16 term names from the distance channels", NAMES, NAMES.replace('names["line"]', 'names["dist"]')),
    ("M17 no clamp in the denominator (0/0)", RET, RET.replace(".clamp(min=EPS)", "")),
    ("M18 w only on the positives", WW, "ww = w * pos * pos_m + neg_m"),
    ("M19 normaliser summed over channels too", RET, RET.replace("ww.sum((0, 2, 3))", "ww.sum()")),
    ("M20 detach the logits (cut gradient)", RET, RET.replace("(logits, target, reduction", "(logits.detach(), target, reduction")),
    ("M21 ignore compared in cells (x0.5)", DIST, DIST.replace(">= ignore", ">= ignore * 0.5")),
    ("M22 normalise per frame (mean over B of per-frame ratios)", RET, 'return (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * ww).sum((2, 3)).div(ww.sum((2, 3)).clamp(min=EPS)).mean(0)'),
    ("M23 BCE target = positive mask (equivalent on binary GT)", RET, RET.replace("(logits, target, reduction", "(logits, pos_m, reduction")),
    ("M24 positive weight pos+1", WW, "ww = w * ((pos + 1.0) * pos_m + neg_m)"),
    ("M25 negatives weight 2", WW, "ww = w * (pos * pos_m + 2.0 * neg_m)"),
    ("M26 default path: lines group + 1e-4 (a regression of old runs)", G0, G0 + " + 1e-4"),
    ("M27 default path: huber term x 1.000001 (old runs)", G0, "groups[\"lines\"] = dice.mean() + 1.000001 * hub.mean()"),
]

orig = open(REAL, encoding="utf-8", newline="").read()
only = sys.argv[2:] if len(sys.argv) > 2 else None
try:
    for name, old, new in M:
        if only and not any(name.startswith(o + " ") or name.startswith(o) and name[len(o)] == " " for o in only):
            continue
        assert orig.count(old) == 1, (name, orig.count(old))
        open(COPY, "w", encoding="utf-8", newline="").write(orig.replace(old, new))
        r = subprocess.run([sys.executable, TEST], cwd=HERE, capture_output=True, text=True, timeout=300,
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2"))
        fails = [l.strip()[4:].strip() for l in r.stdout.splitlines() if l.startswith("  FAIL")]
        crash = r.returncode != 0 and not fails
        last = (r.stderr.strip().splitlines() or [""])[-1]
        print(f"{name:66s} {'CAUGHT  ' if r.returncode else 'SURVIVED'} rc={r.returncode} fails={len(fails)}{' CRASH: ' + last[:90] if crash else ''}", flush=True)
        for f in fails[:2]:
            print("      - " + f[:120])
finally:
    open(COPY, "w", encoding="utf-8", newline="").write(orig)
