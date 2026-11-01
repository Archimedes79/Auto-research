"""build DECISIONS.md / grants.csv blobs = HEAD + only this session's rows (other chats' uncommitted rows stay out); run from the repo root in WSL"""
import re
import subprocess

S = "/mnt/c/Users/mmeut/AppData/Local/Temp/claude/C--Auto-research/6e9638e0-a42f-4660-b7d5-cac910d9084d/scratchpad/"
D = "bev_fusion_lane_center/experiment_execution/DECISIONS.md"
G = "bev_fusion_lane_center/experiment_execution/grants.csv"
head = lambda p: subprocess.run(["git", "show", f"HEAD:{p}"], capture_output=True, text=True, check=True).stdout
work = open(D, encoding="utf-8").read()
h = head(D)
for tag in ["| D4 (Mirko 2026-11-01: \"On D4, we need to discuss more\") |",
            "| D17 (Mirko 2026-11-01: \"When a run has finished successfully that is a final run, but here we neede to discuss more\") |",
            "| D18 (decided 2026-11-01, see below) |", "| D19 (decided 2026-11-01, see below) |",
            "| D25 (decided 2026-11-01, see below) |", "| D33 (decided 2026-11-01, see below) |"]:
    assert tag in work, tag
    plain = "| " + re.match(r"\| (D\d+)", tag).group(1) + " |"
    assert h.count(plain) == 1, plain
    h = h.replace(plain, tag, 1)
mine = [ln for ln in work.splitlines() if re.match(r"^\| 2026-11-01 \| D(25|33|18|19) \(Mirko", ln)]
assert len(mine) == 4, len(mine)
hdr = "| date | decision | consequence |\n|---|---|---|\n"
assert hdr in h
h = h.replace(hdr, hdr + "\n".join(mine) + "\n", 1)
open(S + "DECISIONS.md.blob", "w", encoding="utf-8").write(h)
g = head(G)
add = [ln for ln in open(G, encoding="utf-8").read().splitlines() if ln.startswith(("2026-11-01T08:57,", "2026-11-01T08:58,"))]
assert len(add) == 2
open(S + "grants.csv.blob", "w", encoding="utf-8").write(g + "\n".join(add) + "\n")
print("ok", len(mine), len(add))
