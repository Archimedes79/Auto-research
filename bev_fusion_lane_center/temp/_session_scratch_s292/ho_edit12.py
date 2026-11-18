import io
p = "HANDOVER.md"
s = io.open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert a in s, a
    s = s.replace(a, b, 1)
rep("(drafted, gate-checked: temp/_session_scratch_s292/e343_row.txt) + grant + ONE run (R1 - R3 confirmed, D95)", "(drafted: temp/_session_scratch_s292/e343_row.txt) + grant + ONE run (D95)")
rep("**E345** probe committed (fd40869a); chain `temp/day21/day21.sh` since 22:10 (GPU sweeps, CPU scoring, bootstraps, table; markers `temp/day21/*.done`; results in the card folder); B side = later card.", "**E345** probe committed; chain day21 (markers `temp/day21/*.done`).")
rep("**E344** plan revised after its critic; Q0 = the lead's card E346, E347 beside it: plans until Mirko says build.", "**E344** plan; E346 / E347 = plans until Mirko says build.")
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
