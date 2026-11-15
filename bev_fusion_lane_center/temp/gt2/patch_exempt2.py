p = "evaluation/qgraph_readout.py"
s = open(p, encoding="utf-8").read()
s = s.replace('(?:s(\\d+(?:\\.\\d+)?))?)(x)?@(\\d+(?:\\.\\d+)?)$")', '(?:s(\\d+(?:\\.\\d+)?))?)(x(?:(\\d+(?:\\.\\d+)?))?)?@(\\d+(?:\\.\\d+)?)$")')
s = s.replace("float(m.group(6)), float(m.group(4)) if m.group(4) else None, bool(m.group(5)))", "float(m.group(7)), float(m.group(4)) if m.group(4) else None, (float(m.group(6)) if m.group(6) else 0.5) if m.group(5) else False)")
s = s.replace('Spec = namedtuple("Spec", "kind radius thr sigma exempt", defaults=(None, False))', 'Spec = namedtuple("Spec", "kind radius thr sigma exempt", defaults=(None, False))        # exempt: False, or the flag probability from which a stop / exit / split / merge node is protected (x = 0.5, x0.8 = 0.8)')
s = s.replace('("x" if spec.exempt else "")', '(("x" if spec.exempt == 0.5 else f"x{spec.exempt:g}") if spec.exempt else "")')
s = s.replace("protect = (np.asarray(nd[\"flags\"], float).reshape(len(e.reshape(-1)), -1)[:, :4] >= 0.5).any(1)", "protect = (np.asarray(nd[\"flags\"], float).reshape(len(e.reshape(-1)), -1)[:, :4] >= float(spec.exempt)).any(1)")
s = s.replace("(x = the nodes of the special types stop / exit / split / merge at p >= 0.5 are never suppressed)", "(x[<p>] = the nodes of the special types stop / exit / split / merge at p >= 0.5 (or <p>) are never suppressed)")
open(p, "w", encoding="utf-8").write(s)
t = "temp/gt2/t_exempt.py"
u = open(t, encoding="utf-8").read()
u = u.replace('print("ok")', 'sp = QR.parse("close:1.5x0.95@0.5"); assert sp.exempt == 0.95 and QR.name(sp) == "close1.5x0.95_t0.5", sp\nassert QR.apply(pred, sp)["nodes"]["e"][1] == 0.0          # the exit flag is 0.9 < 0.95: not protected\nassert QR.parse("close:1.5x@0.5").exempt == 0.5 and QR.name(QR.parse("close:1.5x@0.5")) == "close1.5x_t0.5"\nprint("ok")')
open(t, "w", encoding="utf-8").write(u)
print("patched")
