import io
p = "HANDOVER.md"
s = io.open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert a in s, a
    s = s.replace(a, b, 1)
rep("2026-11-15 00:05", "2026-11-15 02:10")
rep("**E342** part A installed (27fee60b, 21:48; runs started before hold the old code), part B (idle-GPU profile, new baseline about 15.7 ms) open.",
    "**E334** NOT adopted (card section 2: one guard missed by 0.0004, topo F1 -0.001 .. -0.005, pair precision lower; no re-run). **E342** A installed (27fee60b), B (idle-GPU profile) open.")
rep("transfer = card E348 (critic, smoke ok), chain day24 queued.", "transfer = card E348 (smoke ok), chain day24 queued.")
rep("**Next:** E343 code + checks; E345 results; E348 critic + start script; old-chain step 1 (D97); dead-code batches; results E318, E319, E334; E349 critic + build; E350 card (E325 head on E324 s).",
    "**Next:** E343 code + checks; E345 results; old-chain step 1 (D97); dead-code batches; results E318, E319, E348; E349 critic + build; E350 card (E325 head on E324 s).")
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
