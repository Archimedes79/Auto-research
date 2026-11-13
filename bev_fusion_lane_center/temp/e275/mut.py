"""one-off (E275): mutation check of the new tests. python mut.py <readout|verdict|glue> <id>...; applies one string replacement to the module source, runs the real test's main(); prints CAUGHT / SURVIVED"""
import contextlib
import importlib.util
import io
import os
import sys
import types

EV = os.path.join(os.environ["BFLC_ROOT"], "evaluation")
RO = {
    "R1": ("< r2:", "<= r2:"),
    "R2": ("cand = np.flatnonzero(e >= thr)", "cand = np.flatnonzero(e > thr)"),
    "R3": ('np.argsort(-e[cand], kind="stable")', "np.argsort(-e[cand])"),
    "R4": ('order = cand[np.argsort(-e[cand], kind="stable")]', 'order = cand[::-1][np.argsort(-e[cand][::-1], kind="stable")]'),
    "R5": ("np.where(keep, e.reshape(-1), 0.0)", "np.where(keep | (e.reshape(-1) < spec.thr), e.reshape(-1), 0.0)"),
    "R6": ('(np.asarray(apply(p, spec)["nodes"]["e"], float) >= spec.thr)', '(np.asarray(apply(p, spec)["nodes"]["e"], float) > spec.thr)'),
    "R7": ("1000.0 * (time.perf_counter() - t0) / max(len(preds), 1)", "1000.0 * (time.perf_counter() - t0)"),
    "R8": ("kept[:n] - pos[i]", "kept[n-1:n] - pos[i]"),
    "R9": ("q = copy.copy(pred)", "q = pred"),
    "R10": ('sum(int((np.asarray(p["nodes"]["e"], float) >= spec.thr).sum()) for p in preds)', 'sum(int((np.asarray(p["nodes"]["e"], float) > spec.thr).sum()) for p in preds)'),
    "R11": ("if not spec.thr > 0.0:", "if False:"),
    "R12": ("assert np.isfinite(pos[cand]).all()", "assert True or np.isfinite(pos[cand]).all()"),
    "R13": ("n += int(((d2 < spec.radius ** 2).sum() - len(g)) // 2)", "n += int(((d2 < spec.radius ** 2).sum() - len(g)))"),
    "R14": ("out.append(dict(row=int(r), idx=k, e=e[k],", "out.append(dict(row=int(r), idx=k, e=e[k].astype(np.float32),"),
    "R15": ("if n and np.min(((kept[:n] - pos[i]) ** 2).sum(1)) < r2:", "if n and np.min(((kept[:n] - pos[i]) ** 2).sum(1)) < r2 and e[i] < 0.95:"),
}
QR = {
    "V1": ('if "down" in sides or not all(guards.values()):', "if not all(guards.values()):"),
    "V2": ('guards["lanes_found_not_below_control_by_%g" % lanes_tol] = all(x >= -lanes_tol for x in lanes)', 'guards["lanes_found_not_below_control_by_%g" % lanes_tol] = (sum(lanes) / len(lanes) >= -lanes_tol)'),
    "V3": ('"junction_F1", 0.03, root, verbose=False, topo_guard=False)', '"junction_F1", 0.02, root, verbose=False, topo_guard=False)'),
    "V4": ('"topo_F1", 0.02, root, verbose=False, topo_guard=False)', '"topo_F1", 0.03, root, verbose=False, topo_guard=False)'),
    "V5": None,
    "V6": ('mean(lambda c: c["diff"]["exact"]["mean"]) >= -0.01', 'min(c["diff"]["exact"]["mean"] for c in cv) >= -0.01'),
    "V7": ("F1_1_not_below_control_by_0_01=all(g >= -0.01 for g in f1)", "F1_1_not_below_control_by_0_01=(sum(f1) / len(f1) >= -0.01)"),
    "V8": ('"junction_F1", 0.03, root, verbose=False, topo_guard=False)', '"junction_F1", 0.03, root, verbose=False, topo_guard=True)'),
    "V10": ("all(x >= -lanes_tol for x in lanes)", "all(abs(x) <= lanes_tol for x in lanes)"),
    "V11": ('mean(lambda c: c["diff"]["error"]["mean"]) >= 0.03', 'min(c["diff"]["error"]["mean"] for c in cv) >= 0.03'),
    "V12": ('for a, b in zip(cv, cc)]\n    guards = dict(t["guards"])', 'for a, b in zip(cv, cc[::-1])]\n    guards = dict(t["guards"])'),
    "V13": ('elif sides == ("inside", "inside"):', 'elif "inside" in sides and "up" not in sides:'),
    "V14": ('side = ("up" if all(g > band for g in gain)', 'side = ("up" if any(g > band for g in gain)'),
    "V15": ('"up" if all(g > band for g in gain) else "down" if any(g < -band for g in gain)', '"up" if all(g > band for g in gain) else "down" if all(g < -band for g in gain)'),
    "V16": ('assert all(c["frames"] == frames for c in cv + cc), ', "assert True or "),
    "V17": ('assert all(c.get("readout", {}).get("spec") == spec for c in cv), ', "assert True or "),
    "V18": ('assert all(c.get("readout", {}).get("spec", "plain_t0.5") == "plain_t0.5" for c in cc), ', "assert True or "),
    "V19": ('[c["seed"] for c in cv] == [c["seed"] for c in cc] == [0, 1], ', '[c["seed"] for c in cv] == [c["seed"] for c in cc], '),
    "V20": ("elif sides == (\"up\", \"up\"):", "elif sides[0] == \"up\":"),
}
GL = {
    "G1": ("if s.thr >= 0.5:\n                        raise SystemExit", "if s.thr >= 0.9:\n                        raise SystemExit"),
    "G2": ('if max(dif.values()) > repro_tol:', 'if max(dif.values()) > 1.0:'),
    "G3": ('f"{run}.{RO.name(s)}" if readouts else run', "run"),
    "G4": ("del preds", "pass"),
    "G5": ('c["readout"] = {**dict(spec=RO.name(s), kind=s.kind, radius=s.radius, thr_e=s.thr), **(info.get(s) or {})}', "c['readout'] = dict(spec='x')"),
    "G6": ("out, rec = QE.score_frames([RO.apply(p, s) for p in preds], frames, thr_e=s.thr, records=True)", "out, rec = QE.score_frames([RO.apply(p, s) for p in preds], frames, thr_e=0.5, records=True)"),
    "G7": ('if dropped_frac_placeholder', 'if dropped_frac_placeholder'),
}


def load(name, path, repl):
    src = open(path, encoding="utf-8").read()
    if repl:
        assert repl[0] in src, f"pattern not found {repl[0]!r}"
        src = src.replace(repl[0], repl[1])
    m = types.ModuleType(name)
    m.__file__ = path
    sys.modules[name] = m
    exec(compile(src, path, "exec"), m.__dict__)
    return m


sys.path.insert(0, EV)
sys.path.insert(0, os.path.join(EV, "..", "training", "grid"))
sys.path.insert(0, os.path.join(EV, "tests"))
kind = sys.argv[1]
for mid in sys.argv[2:]:
    for mod in ("qgraph_readout", "qgraph_run_eval", "test_qgraph_readout", "test_verdict275", "test_qgraph_score_glue", "qgraph_eval"):
        sys.modules.pop(mod, None)
    if kind == "readout":
        load("qgraph_readout", f"{EV}/qgraph_readout.py", RO.get(mid))
        tp = f"{EV}/tests/test_qgraph_readout.py"
    else:
        src = open(f"{EV}/qgraph_run_eval.py", encoding="utf-8").read()
        if mid == "V5":
            src = src.replace('"topo_F1", 0.02, root, verbose=False, topo_guard=False)', '"topo_F1", 0.03, root, verbose=False, topo_guard=False)').replace('"junction_F1", 0.03, root, verbose=False, topo_guard=False)', '"junction_F1", 0.02, root, verbose=False, topo_guard=False)')
        else:
            repl = (QR if kind == "verdict" else GL)[mid]
            assert repl[0] in src, f"pattern not found {repl[0]!r}"
            src = src.replace(repl[0], repl[1])
        import qgraph_readout  # noqa: F401  (the real one)
        m = types.ModuleType("qgraph_run_eval")
        m.__file__ = f"{EV}/qgraph_run_eval.py"
        sys.modules["qgraph_run_eval"] = m
        exec(compile(src, m.__file__, "exec"), m.__dict__)
        tp = f"{EV}/tests/{'test_verdict275' if kind == 'verdict' else 'test_qgraph_score_glue'}.py"
    spec = importlib.util.spec_from_file_location("t_mod", tp)
    t = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(t)
        with contextlib.redirect_stdout(io.StringIO()):
            t.main()
        print(mid, "SURVIVED", flush=True)
    except BaseException as e:  # noqa: BLE001
        print(mid, "CAUGHT", type(e).__name__, str(e)[:100].replace("\n", " "), flush=True)
