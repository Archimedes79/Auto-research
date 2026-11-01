"""one-off: HEAD version of a README with only my rows replaced by the working-tree rows (the other hunks of the working file belong to
other chats). Each row must differ from HEAD's row by my substitutions alone (asserted). usage: make_readmes.py <readme> <out>"""
import subprocess
import sys

readme, out = sys.argv[1], sys.argv[2]
SUBS = {
    "bev_fusion/training/README.md": {
        "| `precompute.py` |": [("`vis_radar` (the radar fans, D36; placeholder until `autogt.coverage`)",
                                 "`vis_radar` (the radar fans of the calibrated poses, D36; `autogt.coverage.radar_cells`, cached per rig)")],
        "| `nuscenes_dataset.py` |": [("`collate` (padded steps + `steps_raw` per sample), `available_scenes`",
                                      "`collate` (padded steps + `steps_raw` per sample, `radar_poses` (B, 5, 4, 4) of the rig, `gap_steps` per sample: "
                                      "the ego steps to the neighbouring keyframes from the poses), `available_scenes`")],
    },
    "bev_fusion/tests/README.md": {                            # HEAD's row is older than my earlier uncommitted update: the whole working row is mine
        "| `test_slots.py` |": None,
        "| `test_slots_OLD.py` |": [("CPU, about 35 s, no data; in `run_gates.sh` (6d6275a); plus 5 combinations (dual + ego-GRU, CMX + radar off, radar memory + pillars, "
                                 "camera off + gate, bevfusion + zsum + SECOND)",
                                 "CPU, about 100 s, no data; in `run_gates.sh` (6d6275a); plus 7 combinations (dual + ego-GRU, CMX + radar off, radar memory + pillars, "
                                 "camera off + gate, bevfusion + zsum + SECOND, `delta_slots` alone and with the radar memory); each build also runs the precompute "
                                 "step (every map in the contract, shape and range)"),
                                ("state carried; invalid combinations refused by `check_config`; the parts that cannot train",
                                 "state carried; a loss on the dense boxes and soft lines reaches every parameter that a loss on the maps they are built from reaches "
                                 "(the planner's door into the network); invalid combinations refused by `check_config`; the parts that cannot train")],
    },
}[readme]
head = subprocess.run(["git", "show", f"HEAD:{readme}"], capture_output=True, text=True, encoding="utf-8", check=True).stdout.split("\n")
work = open(readme, encoding="utf-8").read().split("\n")
for prefix, subs in SUBS.items():
    if prefix.endswith("_OLD.py` |"):
        continue
    hi = [i for i, l in enumerate(head) if l.startswith(prefix)]
    wi = [l for l in work if l.startswith(prefix)]
    assert len(hi) == 1 and len(wi) == 1, (prefix, hi, len(wi))
    row = head[hi[0]]
    if subs is None:
        head[hi[0]] = wi[0]
        continue
    for old, new in subs:
        assert row.count(old) == 1, (prefix, old[:40])
        row = row.replace(old, new)
    assert row.replace("\\|", "|") == wi[0].replace("\\|", "|"), (prefix, "the working row differs from HEAD's by more than my change")
    head[hi[0]] = wi[0]                                       # (an escaped pipe inside a table cell is the only other difference)
open(out, "w", encoding="utf-8", newline="\n").write("\n".join(head))
print("ok", readme, len(SUBS), "rows")
