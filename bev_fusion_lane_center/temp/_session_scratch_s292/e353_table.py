"""E353 result tables from the cards of the run folders (numbers never typed): per seed and read-out the rows c = E324 s, cp (continuation), n1 (narrowed + fine-tuned), n1z (narrowed, no fine-tune).
usage (problem folder, WSL): python3 temp/_session_scratch_s292/e353_table.py > temp/e353/table.md"""
import json
import os

R = os.getcwd()
CARDS = [("snapped", "ends_card_snap_thr0.1.json"), ("PRIMARY", "ends_card_snap_nogeom_attach_thr0.1.json"), ("attach-only", "ends_card_attach_thr0.1.json"), ("unsnapped (not judged)", "ends_card_thr0.1.json")]
KEYS = [("topo_F1", 4), ("F1@1.0", 4), ("junction_F1", 4), ("lanes_found", 4), ("dup_frac", 4), ("vert100_dp", 2)]
out = []
for S in (0, 1):
    runs = [("c = E324 s", f"e324_s_s{S}"), ("cp", f"e353_cp_s{S}"), ("n1", f"e353_n1_s{S}"), ("n1z", f"e353_n1z_s{S}")]
    for label, card in CARDS:
        out.append(f"\n**Seed {S}, {label}** (VAL, 873 frames; head numbers)\n\n| run | " + " | ".join(k for k, _ in KEYS) + " |\n|---|" + "---|" * len(KEYS))
        ref = None
        for name, run in runs:
            p = os.path.join(R, "runs", run, card)
            if not os.path.exists(p):
                out.append(f"| {name} | " + " | ".join("-" for _ in KEYS) + " |")
                continue
            h = json.load(open(p))["numbers"]["head"]
            if name == "c = E324 s":
                ref = h
            cells = []
            for k, d in KEYS:
                x = h[k]
                cells.append(f"{x:.{d}f}" + ("" if ref is None or name == "c = E324 s" else f" ({x - ref[k]:+.{d}f})"))
            out.append(f"| {name} | " + " | ".join(cells) + " |")
print("\n".join(out))
for S in (0, 1):
    for run in (f"e324_s_s{S}", f"e353_cp_s{S}", f"e353_n1_s{S}"):
        p = os.path.join(R, "runs", run, "status.json")
        if os.path.exists(p):
            st = json.load(open(p))
            print(f"\n{run} status: " + json.dumps({k: st[k] for k in list(st)[:8]})[:300])
