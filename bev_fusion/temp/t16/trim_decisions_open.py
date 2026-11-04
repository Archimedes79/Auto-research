"""scratch: remove from the Open table of DECISIONS.md the rows whose decision is already recorded (each has its own row in the Decided table starting `| date | D<n>`, which is what the gate hook reads);
D4 (open) and D31 (reopenable) stay. Verifies every removed id against the Decided table first and refuses otherwise.
    wsl python3 bev_fusion/temp/t16/trim_decisions_open.py [--write]     (from the repository root)"""
import re
import sys

P = "bev_fusion_lane_center/experiment_execution/DECISIONS.md"
lines = open(P, encoding="utf-8").read().split("\n")
i_open = next(i for i, l in enumerate(lines) if l.startswith("## Open"))
i_dec = next(i for i, l in enumerate(lines) if l.startswith("## Decided"))
decided_ids = set()
for l in lines[i_dec:]:
    m = re.match(r"^\|\s*[\d/-]+\s*\|\s*(D\d+)\b", l)
    if m:
        decided_ids.add(m.group(1))
keep, drop = [], []
for i, l in enumerate(lines):
    if i_open < i < i_dec and l.startswith("| D"):
        m = re.match(r"^\| (D\d+) \((decided|replaced|settled)", l)
        if m and m.group(1) in decided_ids and m.group(1) not in ("D4", "D31"):
            drop.append((m.group(1), len(l)))
            continue
        if m and m.group(1) not in decided_ids and m.group(2) == "decided":
            sys.exit(f"{m.group(1)} says decided but has no Decided row: refused")
    keep.append(l)
print("removed from Open:", [d for d, _ in drop], sum(n for _, n in drop), "characters")
print("Decided ids found:", len(decided_ids))
out = "\n".join(keep)
note = "Open means: waiting for Mirko's answer. A row leaves this table when his answer is recorded; it then lives only as a row of the Decided table (cleanup of 2026-11-03, D49). D31 stays because it can be reopened."
out = out.replace("## Open\n", "## Open\n\n" + note + "\n", 1) if note not in out else out
if "--write" in sys.argv:
    open(P, "w", encoding="utf-8", newline="").write(out)
    print("written")
