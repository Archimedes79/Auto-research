p = "evaluation/qgraph_readout.py"
s = open(p, encoding="utf-8").read()
s = s.replace('Spec = namedtuple("Spec", "kind radius thr")        # kind "plain" (radius None), "close" or "mnms" (E288: one-pass Matrix-NMS decay)',
              'Spec = namedtuple("Spec", "kind radius thr sigma", defaults=(None,))        # kind "plain" (radius None), "close" or "mnms" (E288: one-pass Matrix-NMS decay, sigma = its Gaussian width, None = 0.5)')
s = s.replace('_SPEC = re.compile(r"^(plain|close:(\\d+(?:\\.\\d+)?)|mnms:(\\d+(?:\\.\\d+)?))@(\\d+(?:\\.\\d+)?)$")', '_SPEC = re.compile(r"^(plain|close:(\\d+(?:\\.\\d+)?)|mnms:(\\d+(?:\\.\\d+)?)(?:s(\\d+(?:\\.\\d+)?))?)@(\\d+(?:\\.\\d+)?)$")')
s = s.replace("""    spec = Spec(kind, None if kind == "plain" else float(m.group(2) if kind == "close" else m.group(3)), float(m.group(4)))""",
              """    spec = Spec(kind, None if kind == "plain" else float(m.group(2) if kind == "close" else m.group(3)), float(m.group(5)), float(m.group(4)) if m.group(4) else None)""")
s = s.replace('''    return f"plain_t{spec.thr:g}" if spec.kind == "plain" else f"{spec.kind}{spec.radius:g}_t{spec.thr:g}"''', '''    if spec.kind == "plain":
        return f"plain_t{spec.thr:g}"
    return f"{spec.kind}{spec.radius:g}" + (f"s{spec.sigma:g}" if spec.sigma else "") + f"_t{spec.thr:g}"''')
s = s.replace("q[\"nodes\"][\"e\"] = matrix_nms(nd[\"pos\"], e, spec.radius, spec.thr).reshape(e.shape)", "q[\"nodes\"][\"e\"] = matrix_nms(nd[\"pos\"], e, spec.radius, spec.thr, spec.sigma or 0.5).reshape(e.shape)")
s = s.replace('expected plain@<thr>, close:<radius>@<thr> or mnms:<radius>@<thr>', 'expected plain@<thr>, close:<radius>@<thr> or mnms:<radius>[s<sigma>]@<thr>')
open(p, "w", encoding="utf-8").write(s)
print("ok")
