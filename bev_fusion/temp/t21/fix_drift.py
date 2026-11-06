p = "temp/t21/stream_drift.py"
s = open(p, encoding="utf-8").read()

def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (s.count(a), a)
    s = s.replace(a, b)

rep("    state, prev_state, rows = None, None, []", "    state, rows = None, []")
rep("                prev_state, state = state, m.advance(to(one(f)), state, step)\n", "                state = m.advance(to(one(f)), state, step)\n")
rep("dts=(0.0,), state=prev_state, step=step) if t >= 2 else None", "dts=(0.0,), state=state, step=step) if t >= 2 else None")
rep("            prev_state, state = state, res[\"state\"]\n", "            state = res[\"state\"]\n")
rep("pair = the batch of the two last frames given the stream's own state of the frame before", "pair = the batch of the two last frames given the stream's own state of the frame before (the memory carried)")
open(p, "w", encoding="utf-8", newline="").write(s)
