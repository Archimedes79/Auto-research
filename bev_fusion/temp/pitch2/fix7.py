p = "../bev_fusion/temp/pitch2/patch_round7.py"
s = open(p, encoding="utf-8").read()
a = s.index("    ('''the Evaluation is the one front door and calls it; `tap`")
b = s.index("    (\"(the health check, runner_api.py).\"")
new = '''    ("""Evaluation is the one front door and calls it; `tap`, `time` and `probe` are the Runner's, passed through. Which Runner: the class a project names in `evaluation/evalsuite.json` ("runner":
"module:Class", BEV: runner.runner:Runner, whose command line is `python -m runner`), else the generic one of runner_api.py; `Evaluation(runner=...)` injects another. Everything else here reads
what was stored (only `draw_frame`, which draws a frame, runs the network on it again).""", """Evaluation is the one front door and ALWAYS includes the Runner of its project: the class it names in `evaluation/evalsuite.json` ("runner": "module:Class", BEV: runner.runner:Runner, whose
command line is `python -m runner`), built with (project, adapter); evalsuite has none of its own (a project that names none is an error; `Evaluation(runner=...)` injects one, the toy selftest does).
`tap`, `time` and `probe` are the Runner's, passed through. Everything else here reads what was stored (only `draw_frame`, which draws a frame, runs the network on it again)."""),
'''
s = s[:a] + new + s[b:]
open(p, "w", encoding="utf-8").write(s)
print("fixed")
