import sys
def edit(s):
    lines = s.split("\n")
    lines = [l for l in lines if not l.startswith("| `evaluate_run.py` | DEPRECATED forwarding stub")]
    s = "\n".join(lines)
    old = "The pre-suite scripts (`evaluate_run.py`, `observe_run.py`, `eval_memory.py`, `video_encode.py`) are in `../_to_delete/evaluation_pre_suite/`."
    new = "The pre-suite scripts (`evaluate_run.py`, `observe_run.py`, `eval_memory.py`, `video_encode.py`) are in `../_to_delete/evaluation_pre_suite/`, since 2026-10-31 with the forwarding stub (`evaluate_run_stub.py`)."
    assert s.count(old) == 1; s = s.replace(old, new)
    old = "python evaluation/bev_suite.py compare runs/a runs/b [--tag _warm] [--n 1000]        # paired scene bootstrap b - a"
    new = "python evaluation/bev_suite.py compare runs/a runs/b [--tag _warm] [--n 1000] [--force]   # paired scene bootstrap b - a; refused when the GT signatures differ (manifest / metrics.json, runs/<name>/gt.json; warned when unknown); legacy records scene_records<tag>_n<N>.pkl only on the same frames"
    assert s.count(old) == 1; s = s.replace(old, new)
    return s
for p in sys.argv[1:]:
    s = open(p, encoding="utf-8").read()
    open(p, "w", encoding="utf-8", newline="").write(edit(s))
print("ok")
