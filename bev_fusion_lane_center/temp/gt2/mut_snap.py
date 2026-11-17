import os, subprocess, sys
src = open("evaluation/qends_eval.py", encoding="utf-8").read()
A = "if d <= WIRE_POS and _ang(te, ta) <= WIRE_ANG and d < d_in:"
B = "if d <= WIRE_POS and _ang(ts, tb) <= WIRE_ANG and d < d_out:"
muts = [("entry: no angle test", A, "if d <= WIRE_POS and d < d_in:"), ("exit: no angle test", B, "if d <= WIRE_POS and d < d_out:"),
        ("entry: first piece in list", A, "if d <= WIRE_POS and _ang(te, ta) <= WIRE_ANG and d_in == np.inf:"), ("exit: first piece in list", B, "if d <= WIRE_POS and _ang(ts, tb) <= WIRE_ANG and d_out == np.inf:"),
        ("entry: farthest piece", A, "if d <= WIRE_POS and _ang(te, ta) <= WIRE_ANG and (d_in == np.inf or d > d_in):"),
        ("wire every piece (snap_all always)", "        if not default_wire:\n            if w_in is not None:", "        if False:\n            if w_in is not None:"),
        ("guard never reverts", "if np.linalg.norm(ch) < 1.0 or ch @ ta <= 0 or ch @ tb <= 0:", "if False:"),
        ("dedup never skips", "if (w_in, w_out) in seen:", "if False:"),
        ("snap fields swapped", "pa, ta = ends[w_in][0], ends[w_in][1]", "pa, ta = ends[w_in][2], ends[w_in][3]")]
for name, old, new in muts:
    assert old in src, name
    open("evaluation/_qe_mut.py", "w", encoding="utf-8").write(src.replace(old, new, 1))
    r = subprocess.run([sys.executable, "evaluation/_qe_mut.py", "selftest"], capture_output=True, text=True)
    print(f"{name:36s}", "CAUGHT" if r.returncode != 0 else "NOT CAUGHT")
os.remove("evaluation/_qe_mut.py")
