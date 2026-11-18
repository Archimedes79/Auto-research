"""Ready-to-use prototype (D28): the delivered lane graph of a registry system on chosen frames, written planner-ready, with its numbers and an optional video.

    wsl bash tools/env/_py14.sh lane_prototype.py [name] [--split val] [--scenes 23 scene-0003 ...] [--frames N] [--out dir] [--video] [--fresh]
    wsl bash tools/env/_py14.sh lane_prototype.py recall_dp --scenes 23 7 --video     # two VAL scenes + results video
    wsl bash tools/env/_py14.sh lane_prototype.py recall_dp                           # the whole VAL split: reproduces the stored card (layer 1)

name: a builder entry of models/registry.json (default recall_dp). --scenes: scene ids of data/splits.json / poses (an integer) or nuScenes names (scene-0003);
default all scenes of the split. --frames N: the first N frames of the selection (time order inside a scene). Output (default runs/prototype/<name>/):
scenes/<scene name>.json (format in tools/prototype/README.md), summary.json + summary.md, and with --video <name>_<split>.mp4 + poster png.
The graphs are those evaluate.py scores: the cached raw network outputs (CPU only; --fresh re-runs the network, CUDA only) assembled with the registry
settings, then the registry's output stage (dp_eps). The numbers are sparse_metrics.card_any on exactly the chosen frames (the call of evaluate.py);
without --scenes / --frames they must reproduce runs/eval/<name>/<split>.json (exit 1 otherwise). ms per frame is read from the stored timings of
evaluation/drivers/p18_latency.py (runs/e18/latency/), never measured here. Selftest: test_lane_prototype.py.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "2")  # a shared machine: at most 2 threads
import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import evaluate as EV  # noqa: E402
import make_results_video as RV  # noqa: E402
import sparse_metrics as SM  # noqa: E402

HEAD = ("Q", "topo_F1", "lanes_found", "R@1_far", "vert100")
FRAME = "ego frame of the keyframe, metres, x forward, y left (planner convention; the metric code uses x right, y forward)"
DEC = 5  # decimals of the written coordinates (10 micrometres: the JSON re-scores to the same numbers)


def to_planner(q):
    """polyline [n, 2] of the graph contract (x right, y forward) -> planner frame (x forward, y left)"""
    q = np.asarray(q, np.float64).reshape(-1, 2)
    return np.stack([q[:, 1], -q[:, 0]], 1)


def from_planner(q):
    q = np.asarray(q, np.float64).reshape(-1, 2)
    return np.stack([-q[:, 1], q[:, 0]], 1)


def scene_id(ft, s):
    """scene id of an argument: an integer is the id, anything else a nuScenes scene name"""
    ids = {int(x) for x in ft["scene"]}
    names = {i["scene_name"]: int(sc) for i, sc in zip(ft["info"], ft["scene"])}
    sid = int(s) if str(s).isdigit() else names.get(str(s))
    if sid not in ids:
        sys.exit(f"unknown scene {s}: ids {sorted(ids)}, names {sorted(names)}")
    return sid


def select(ft, scenes=None, frames=0):
    """frame positions of the split: the scenes in the given order (default all, by id), time order inside a scene, the first `frames` of them (0 = all)"""
    ids = [scene_id(ft, s) for s in scenes] if scenes else sorted({int(x) for x in ft["scene"]})
    ks = [int(i) for s in ids for i in sorted(np.nonzero(ft["scene"] == s)[0], key=lambda i: ft["ts"][i])]
    return ks[:frames] if frames else ks


def delivered(e, split, ks, fresh=False):
    """(graphs, per-lane confidences, ground truths, info) of the frames ks: the assembly of evaluate.graphs_of with aux (the source node of every vertex),
    then the registry's output stage dp_eps; confidence of a lane = mean path validity pi of the network nodes its vertices come from"""
    import p16_infer as PI

    raw, gts, info, base = EV.raw_of(e, split, ks, fresh, require_cache=not fresh)  # builder or union (the members' joined trees)
    src = info["cache"]
    cfg, asm = EV.builder_cfg(base), dict(e["asm"])
    eps = asm.pop("dp_eps", 0.0)
    g, conf = [], []
    t0 = time.time()
    for r in raw:
        g0, ax = PI.assemble(r, None, e["tau"], cfg, aux=True, **asm)
        g.append(PI.simplify_dp(g0, eps) if eps > 0 else g0)  # simplify_dp keeps the lanes, their order and their end / junction vertices
        pis = [[float(r["pi"][s]) for s in (ax["src"][v] for v in nd) if s >= 0] for nd in g0["nodes"]]
        conf.append([round(float(np.mean(p)), 4) if p else None for p in pis])
    return g, conf, gts, dict(source=src, asm_ms_cpu=(time.time() - t0) / max(len(gts), 1) * 1e3)


def frame_json(k, info, g, conf):
    lanes = [dict(id=i, points=np.round(to_planner(q), DEC).tolist(), vertex_ids=[int(v) for v in nd], confidence=c)
             for i, (q, nd, c) in enumerate(zip(g["pts"], g["nodes"], conf))]
    return dict(k=int(k), sample_token=info["sample_token"], timestamp_us=int(info["timestamp"]), lanes=lanes, edges=[[int(a), int(b)] for a, b in g["edges"]], route=None)


def graph_of_frame(f):
    """a frame of the written JSON -> graph contract (x right, y forward): what the selftest re-scores"""
    return dict(pts=[from_planner(ln["points"]) for ln in f["lanes"]], edges=[tuple(x) for x in f["edges"]], nodes=[ln["vertex_ids"] for ln in f["lanes"]])


def stored_timing(e, temporal):
    """ms per frame of the chain (builder, assembly, route input, route head, consumer; batch 1, cached features, idle GPU) from the latest p18_latency.py
    session that timed this run (a union: its two members, --name2) with the registry's M, fork pruning and union; when that session ran without the
    time recursion the system uses, the recursion cost of the same session (another run timed with and without) is given next to it. lane_graph_ms =
    builder + assembly of the same session (the lane graph without the route head and the consumer). An entry with other network settings than M / K0
    (e.g. lattice_ring) was never timed as such: it gets no number, only the plain configuration's as `nearest`"""
    reg = EV.registry()
    builders = [reg[m] for m in e["members"]] if e["kind"] == "union" else [e]
    out = _timing([b["run"] for b in builders], builders[0]["set"]["M"], e["asm"], temporal)
    other = sorted({k for b in builders for k in b["set"] if k not in ("M", "K0")})
    if not other:
        return out
    if out.get("ms") is None:
        ms = "not measured either"
    else:
        ms = f"{out['ms']} ms measured {'with' if out['time_recursion_in_timing'] else 'without'} the time recursion"
        ms += f", about {out['ms_estimate_with_recursion']} ms with it" if "ms_estimate_with_recursion" in out else ""
    return dict(ms=None, note=f"not measured for this configuration ({', '.join(other)}); nearest: the plain configuration (M / K0 only), {ms}", nearest=out)


def session_tag(stem, name, name2):
    """the session label of a latency file name (the --tag without its replicate suffix _r<k>): the settings prefix is cut, and since 2026-11-01 the builder names
    (`__<name>[__<name2>]` behind the settings, the form p18_latency.result_file writes) are cut too; files written before have no names and parse as before"""
    for n in (name, name2):
        if n:
            stem = stem.replace(f"__{n}", "", 1)
    return re.sub(r"_r\d+$", "", re.sub(r"^latency_M\d+_rec\d_u\d_fp[\d.]+(_ens2)?_?", "", stem))


def _timing(runs, M, asm, temporal):
    """the stored timing of the builder runs (one, or two joined by --name2) at M and the assembly settings asm (see stored_timing)"""
    if len(runs) > 2:
        return dict(ms=None, note="no timing: the latency driver times one builder or two (--name2)")
    rows = []
    for f in glob.glob(f"{EV.R}/runs/e18/latency/latency_*.json"):
        d = json.load(open(f))
        tag = session_tag(os.path.basename(f)[:-5], d.get("name"), d.get("name2"))
        rows.append(dict(file=os.path.relpath(f, EV.R), mtime=os.path.getmtime(f), tag=tag, name=d.get("name"), name2=d.get("name2") or "", M=d.get("M"), rec=int(d.get("rec", 0)),
                         union=d.get("union"), fp=float(d.get("fork_prune", 0.0)), ms=d["total"]["median_ms"], lane_ms=d["builder"]["median_ms"] + d["assembly"]["median_ms"]))
    want2 = runs[1] if len(runs) == 2 else ""
    mine = [r for r in rows if r["name"] == runs[0] and r["name2"] == want2 and r["M"] == M and r["union"] == asm.get("fast") and abs(r["fp"] - asm.get("fork_prune", 0.0)) < 1e-9]
    if not mine:
        return dict(ms=None, note="no stored timing of this run with these settings: time it with evaluation/drivers/p18_latency.py on an idle GPU")
    same = [r for r in mine if r["rec"] == int(temporal)] or mine
    last = max(same, key=lambda r: r["mtime"])["tag"]
    sess = [r for r in same if r["tag"] == last]
    out = dict(ms=round(float(np.median([r["ms"] for r in sess])), 2), lane_graph_ms=round(float(np.median([r["lane_ms"] for r in sess])), 2),
               replicates=len(sess), session=last, files=sorted(r["file"] for r in sess),
               time_recursion_in_timing=bool(sess[0]["rec"]), time_recursion_in_system=bool(temporal),
               source="evaluation/drivers/p18_latency.py (median of the per-replicate medians of 'total'; the output stage dp_eps is not in the timed chain)")
    if sess[0]["rec"] != int(temporal):
        exp = last.split("_")[0]
        pair = {}
        for r in rows:
            if r["tag"].split("_")[0] == exp and not r["name2"]:
                pair.setdefault((r["name"], r["M"], r["fp"], r["union"]), {}).setdefault(r["rec"], []).append(r["ms"])
        dif = [(float(np.median(v[1])) - float(np.median(v[0])), k[0]) for k, v in sorted(pair.items(), key=str) if 0 in v and 1 in v]
        if dif:
            rec = dif[0][0] * len(runs)  # once per builder
            out["time_recursion_ms"] = round(rec, 2)
            out["time_recursion_measured_on"] = dif[0][1] + (f" (x {len(runs)} builders)" if len(runs) > 1 else "")
            out["ms_estimate_with_recursion"] = round(out["ms"] + rec, 2)
            out["lane_graph_ms_estimate_with_recursion"] = round(out["lane_graph_ms"] + rec, 2)
    return out


def git_head():
    """short commit of the checkout without the dirty check of evaluate.commit (git status on /mnt/c takes minutes on a loaded machine)"""
    try:
        return subprocess.run(["git", "-c", "safe.directory=*", "-C", EV.R, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=60).stdout.strip() or "no-git"
    except Exception:  # noqa: BLE001
        return "no-git"


def check_card(name, split, m):
    """layer 1: the whole split must reproduce the stored card of evaluate.py"""
    p = f"{EV.R}/runs/eval/{name}/{split}.json"
    if not os.path.exists(p):
        return dict(card=os.path.relpath(p, EV.R), reproduced=None, note="no stored card")
    st = json.load(open(p, encoding="utf-8-sig"))
    diff = {k: round(m[k] - st["metrics"][k], 6) for k in EV.GOLD_KEYS if k in st["metrics"] and k in m}
    ok = st["frames"] == m["n_samples"] and all(abs(v) <= EV.TOL for v in diff.values())
    return dict(card=os.path.relpath(p, EV.R), reproduced=bool(ok), diff=diff, tol=EV.TOL)


def summary_md(s):
    m, t = s["metrics"], s["timing"]
    lines = [f"# Lane prototype: {s['system']} on {s['split'].upper()}", "",
             f"{s['frames']} frames of {len(s['scenes'])} scene(s) ({', '.join(x['name'] for x in s['scenes'][:6])}{' ...' if len(s['scenes']) > 6 else ''}); "
             f"graphs from {'the network (fresh)' if s['fresh'] else 'the cached network outputs'}, assembled with the registry settings; {s['date']}, commit {s['commit']}.", "",
             "| " + " | ".join(HEAD) + " | ms per frame |", "|" + "---|" * (len(HEAD) + 1),
             "| " + " | ".join(f"{m[k]:.3f}" if k in m else "-" for k in HEAD) + f" | {t['ms'] if t.get('ms') is not None else '-'} |", ""]
    if t.get("ms") is not None:
        lines.append(f"ms per frame: {t['ms']} ms, stored timing ({t['replicates']} replicates, session {t['session']}: {t['files'][0]} ...), {t['source']}.")
        est = "ms_estimate_with_recursion" in t
        if est:
            lines.append(f"The timing ran without the time recursion that this system uses; the recursion cost {t['time_recursion_ms']} ms in the same session "
                         f"(on {t['time_recursion_measured_on']}), so about {t['ms_estimate_with_recursion']} ms per frame with it.")
        if est:  # the label says what the number is: measured with / without the recursion, or the estimate
            rec = " with the time recursion (estimate: measured without it + its cost)"
        elif t["time_recursion_in_timing"]:
            rec = " with the time recursion" + ("" if t["time_recursion_in_system"] else " (measured with it; this system runs without it)")
        else:
            rec = " without the time recursion" + (" (this system uses it; its cost was not measured in this session)" if t["time_recursion_in_system"] else "")
        lines +=["", "| estimate | ms per frame |", "|---|---|",
                  f"| lane graph = builder + assembly (no route head, no consumer){rec} | {t['lane_graph_ms_estimate_with_recursion'] if est else t['lane_graph_ms']} |",
                  f"| full chain (builder, assembly, route input, route head, consumer){rec} | {t['ms_estimate_with_recursion'] if est else t['ms']} |", ""]
    else:
        lines.append(t["note"])
    if s.get("stored_card"):
        c = s["stored_card"]
        lines.append(f"Layer 1: the stored card {c['card']} is {'reproduced' if c['reproduced'] else 'NOT reproduced'} (tolerance {c.get('tol')}).")
    lines += ["", f"Files: {s['out']}/scenes/<scene>.json (planner format: tools/prototype/README.md)" + (f", video {s['video']}" if s.get("video") else "") + "."]
    return "\n".join(lines) + "\n"


def run(name="recall_dp", split="val", scenes=None, frames=0, out=None, video=False, fresh=False):
    """the prototype; returns the summary dict (written as summary.json)"""
    reg = EV.registry()
    if name not in reg or reg[name]["kind"] not in ("builder", "union"):
        sys.exit(f"{name}: not a builder or union entry of models/registry.json (stored-graph references such as the control are not a deliverable system)")
    e = reg[name]
    base = reg[e["members"][0]] if e["kind"] == "union" else e
    if fresh:
        import torch

        if not torch.cuda.is_available():
            sys.exit("--fresh refused: no CUDA GPU. The network path (evaluate.builder_raw) runs on the GPU only; drop --fresh to use the cached network outputs")
    out = os.path.join(EV.R, out or f"runs/prototype/{name}")  # a relative --out is read from the problem folder (_py14.sh runs in the script's folder)
    ft = RV.frame_table(split)
    ks = select(ft, scenes, frames)
    full = not scenes and not frames
    g, conf, gts, info = delivered(e, split, ks, fresh)
    pos = {k: n for n, k in enumerate(ks)}
    shutil.rmtree(f"{out}/scenes", ignore_errors=True)  # this tool's own output of an earlier call
    os.makedirs(f"{out}/scenes", exist_ok=True)
    sc_rows = []
    for sid in dict.fromkeys(int(ft["scene"][k]) for k in ks):
        kk = [k for k in ks if int(ft["scene"][k]) == sid]
        i0 = ft["info"][kk[0]]
        sc = dict(id=sid, name=i0["scene_name"], location=RV.LOCN[int(ft["loc"][kk[0]])], frames=len(kk))
        doc = dict(schema=1, system=name, registry=e, split=split, scene=sc, frame=FRAME,
                   frames=[frame_json(k, ft["info"][k], g[pos[k]], conf[pos[k]]) for k in kk])
        json.dump(doc, open(f"{out}/scenes/{sc['name']}.json", "w", encoding="utf-8"))
        sc_rows.append(sc)
    m = SM.card_any(g, gts, extra=True)
    m = {k: float(v) for k, v in m.items() if not (isinstance(v, float) and v != v)}
    s = dict(schema=1, system=name, registry=e, split=split, frames=len(ks), scenes=sc_rows, fresh=bool(fresh), source=info["source"],
             metrics={k: m[k] for k in HEAD if k in m}, metrics_all=m, assembly_ms_cpu=round(info["asm_ms_cpu"], 2),
             timing=stored_timing(e, EV.builder_cfg(base).temporal), route=None, commit=git_head(), date=time.strftime("%Y-%m-%d %H:%M"), out=os.path.relpath(out, EV.R))
    s["route_note"] = "no route flag: the registry systems end at the lane graph; the route head of the latency chain is not part of evaluate.py's path"
    if full:
        s["stored_card"] = check_card(name, split, m)
    if video:
        lines = [f"these {len(ks)} {split.upper()} frames: Q {m['Q']:.3f}  lanes found {m['lanes_found']:.2f}", f"vertices per 100 m {m['vert100']:.1f}"]
        groups = [(sc["id"], [k for k in ks if int(ft["scene"][k]) == sc["id"]]) for sc in sc_rows]
        gk = {k: g[pos[k]] for k in ks}
        tk = {k: gts[pos[k]] for k in ks}
        s["video"] = os.path.relpath(RV.video(name, gk, tk, ft, groups, lines, f"{out}/{name}_{split}.mp4", split), EV.R)
    json.dump(s, open(f"{out}/summary.json", "w", encoding="utf-8"), indent=1)
    open(f"{out}/summary.md", "w", encoding="utf-8").write(summary_md(s))
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", nargs="?", default="recall_dp")
    ap.add_argument("--split", default="val", choices=list(EV.SPLITS))
    ap.add_argument("--scenes", nargs="*", default=[], help="scene ids (integer) or nuScenes names (scene-0003); default all scenes of the split")
    ap.add_argument("--frames", type=int, default=0, help="the first N frames of the selection (0 = all)")
    ap.add_argument("--out", default=None, help="output folder (default runs/prototype/<name>/)")
    ap.add_argument("--video", action="store_true", help="results video of the chosen frames (camera + graph, prediction red over ground truth green)")
    ap.add_argument("--fresh", action="store_true", help="re-run the network instead of the cache (CUDA only)")
    ap.add_argument("--test-id", default="", help="TEST only with a run id logged in evaluation/TEST_USAGE_LOG.md")
    a = ap.parse_args()
    if a.split == "test" and not EV.test_allowed(a.test_id):
        sys.exit("TEST refused: pass --test-id <run id> that is logged in evaluation/TEST_USAGE_LOG.md (select and tune on VAL only)")
    s = run(a.name, a.split, a.scenes, a.frames, a.out, a.video, a.fresh)
    print(summary_md(s))
    print("written:", f"{s['out']}/summary.json")
    if s.get("stored_card", {}).get("reproduced") is False:
        sys.exit("stored card NOT reproduced: " + json.dumps(s["stored_card"]["diff"]))


if __name__ == "__main__":
    main()
