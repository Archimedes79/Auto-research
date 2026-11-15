import io
p = "HANDOVER.md"
s = io.open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert a in s, a
    s = s.replace(a, b, 1)
rep("VAL rehearsal = day26 (dumps) then day27 (rest, 10 GB gate, RSS in `temp/day27/`), then log row", "CPU rehearsal FAILED the pre-registered tolerance (E312 fresh CPU dumps: vertices, wired / curves counts): GPU branch (card section 2) = chain day28 (job E343.rh_gpu_dumps waits for a GPU slot), TEST line then `--device cuda`; then log row (drafted, gate-checked: temp/_session_scratch_s292/e343_row.txt)")
rep("day26 / day27 = E343 rehearsal.", "day26 / 27 / 28 = E343 rehearsal.")
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
