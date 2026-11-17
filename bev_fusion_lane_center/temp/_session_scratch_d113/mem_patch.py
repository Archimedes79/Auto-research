import io
p = r"/mnt/c/Users/mmeut/.claude/projects/C--Auto-research/memory/lane-d113-token-cache-2026-11-17.md"
s = io.open(p, encoding="utf-8").read()
i = s.index("- **State at 00:35")
j = s.index("- **Numbers (B' seed 0")
new = ("- **State at 01:25 on 2026-11-17: ALL EIGHT caches done** (B' and plain B, seeds 0 / 1, TRAIN + VAL, `runs/e365_cache/`, plus the 40-frame `..._first40` rebuild of the determinism check = `lane_tokcache.py same` -> SAME), the four VAL fixtures PASS bit for bit (B' 246,809 / 255,928, plain B 275,665 / 282,595 nodes; plain B seed 1 has ONE frame whose last rank (B's own top-512 cut) is a tie of two cells with exactly equal existence: `compare_dump` allows exactly that and reports `frames_with_boundary_tie`), commits 3e386bc95e + a727820669, the manifest record is in git (swept into Fable's commit 5e7afefb). Fable's E365 S1 trained on the B' caches and ended MISS (their card).\n")
s = s[:i] + new + s[j:]
io.open(p, "w", encoding="utf-8").write(s)
print("ok")
