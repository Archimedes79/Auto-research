"""scratch (Image Encoder, clean-base round 2026-11-10): config-load check. For every stored config (training/configs/*.json, the configs of the kept / baseline runs in runs/<name>/config.json)
the static start check of training/startable.py (parse, ModelConfig / LossConfig accepted, model.check_config, paths exist) and the result written one line per config:
    python temp/cleanup/t22_config_check.py OUTFILE        (from bev_fusion/, gcar python)
Run it before and after the cleanup and diff the two files; the notes about uncommitted files are left out (they change with the working tree)."""
import glob
import json
import os
import sys

sys.path.insert(0, "training")
import startable  # noqa: E402

KEPT = ["t11_vel_hurdle_acc2", "t10_dual_s0", "t10_dual_s1", "t8_ctl_s0", "t8_ctl_s1", "t20_base_s0", "t21_pair4_s0", "t21_ctl_s0", "t22_B_s0", "t23_C_s0", "t23_C_s1", "t21_ctl_s1"]
paths = sorted(glob.glob("training/configs/*.json")) + [f"runs/{n}/config.json" for n in KEPT if os.path.exists(f"runs/{n}/config.json")]
lines = []
os.makedirs("temp/cleanup/_cc", exist_ok=True)
for p in paths:
    try:
        d = json.load(open(p))
        d.pop("split", None)                         # trainsuite start resolves the split key before the launch: the static check sees the composed form
        q = os.path.abspath("temp/cleanup/_cc/" + p.replace("/", "__"))
        json.dump(d, open(q, "w"))
        problems, notes = startable.check(q, startable.ROOT, False, "")
    except Exception as e:                                                    # noqa: BLE001
        problems, notes = [f"check crashed: {type(e).__name__}: {e}"], []
    notes = [n for n in notes if "uncommitted file" not in n and "git state" not in n]
    lines.append(f"{p}: {'OK' if not problems else 'PROBLEM ' + ' | '.join(problems)[:400]}" + (f" ; notes {len(notes)}" if notes else ""))
open(sys.argv[1], "w").write("\n".join(lines) + "\n")
bad = [ln for ln in lines if ": PROBLEM" in ln]
print(len(lines), "configs checked,", len(bad), "with a problem ->", sys.argv[1])
