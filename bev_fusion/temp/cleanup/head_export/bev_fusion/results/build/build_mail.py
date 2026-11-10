"""The plain-text body of the report mail: the narrative file (cfg.mail.narrative, written by the lead) with the numbers filled in from the stored files.
The narrative holds the prose and these markers, each on a line of its own or inside a sentence:
    @@MAIL_TABLE@@   the table of the runs in cfg.mail.runs (default: the bundle's control and new run) over cfg.mail.rows [[name, key]] (default: the bundle rows)
    @@MAIL_COST@@    ms per frame and the largest modules of those runs (the stored idle-GPU timing; a run without one stops the build)
    @@MAIL_SOA@@     the maps of the re-scoring against GaussianCaR (cfg.inputs.soa_maps[0]) as lines
    @@name@@         any fact of the report (head, front_dual, front_ours, front_diff, ...)
    python bev_fusion/results/build/build_mail.py [--config ...] > mail.txt        (build_report.py writes it into the out folder as well)"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report_lib import fail, fill, md_tables, need  # noqa: E402

SOA_NAMES = {"drivable_area": "drivable area", "walkway": "walkway", "carpark_area": "car park", "ped_crossing": "pedestrian crossing", "lane_divider": "lane divider", "road_divider": "road divider"}


def mail_runs(ctx):
    m = ctx.cfg["mail"]
    ids = m.get("runs") or [ctx.cfg["bundle"]["control"], ctx.cfg["bundle"]["run"]]
    for i in ids:
        if i not in ctx.by_id:
            fail(f"mail: run {i} is not among the runs with an evaluation")
    return [ctx.by_id[i] for i in ids]


def table_lines(ctx):
    runs = mail_runs(ctx)
    rows = ctx.cfg["mail"].get("rows") or [[r["name"], r["key"]] for r in ctx.cfg["bundle"]["rows"] if r.get("key")]
    w, cw = max(len(n) for n, _ in rows) + 2, [max(12, len(r["label"]) + 2) for r in runs]       # a column is as wide as its label needs
    out = [" " * w + "".join(f"{r['label']:>{c}s}" for r, c in zip(runs, cw))]
    for name, key in rows:
        out.append(f"{name:{w}s}" + "".join(f"{ctx.T[r['id']][key]:{c}.3f}" if key in ctx.T[r["id"]] else f"{'-':>{c}s}" for r, c in zip(runs, cw)))
    return out


def cost_lines(ctx):
    out = []
    for r in mail_runs(ctx):
        d = json.load(open(need(f"{r['eval']}/timing_modules.json", f"mail: timing of {r['id']}"), encoding="utf-8"))
        inf = d["inference_ms"]
        mods = sorted(inf["per_frame_ms"].items(), key=lambda kv: -kv[1])[:5]
        out.append(f"  {r['label']:10s} {inf['total_ms']:6.1f} ms per frame ({'idle GPU' if d.get('idle') else 'NOT idle: do not quote'}); largest modules: " + ", ".join(f"{k} {v:.1f}" for k, v in mods))
    return out


def soa_lines(ctx):
    t = md_tables(open(ctx.path("soa_maps"), encoding="utf-8").read())[0][2]
    return [f"  {SOA_NAMES[r[0]]:22s} {r[4]} / {r[5]}   (ours as a share: {r[6]})" for r in t
            if len(r) >= 7 and ((r[0] in ("drivable_area", "walkway", "carpark_area", "ped_crossing") and r[1] == "iou@0") or (r[0] in ("lane_divider", "road_divider") and r[1] == "iou@0.5"))]


def build(ctx):
    text = open(need(ctx.cfg["mail"]["narrative"], "mail narrative"), encoding="utf-8").read()
    blocks = {"MAIL_TABLE": table_lines, "MAIL_COST": cost_lines, "MAIL_SOA": soa_lines}

    def fact(name):
        return "\n".join(blocks[name](ctx)) if name in blocks else ctx.fact(name)

    return fill(text, fact, "mail narrative").rstrip("\n") + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_runs.json"))
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    import build_report as BRP
    cfg = json.load(open(os.path.abspath(a.config), encoding="utf-8"))
    os.chdir(a.root or BRP.PROBLEM)
    sys.stdout.write(build(BRP.Ctx(cfg, cfg["out"])))


if __name__ == "__main__":
    main()
