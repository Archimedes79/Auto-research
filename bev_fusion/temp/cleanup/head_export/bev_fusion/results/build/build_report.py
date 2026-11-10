"""The result report of the BEV system as one html page (images embedded, videos as files next to it), built from the stored evaluation files of the runs.
Promoted from the scratch of the t16 report (v6, 2026-11-03); the same inputs give the same page. What the report contains is in report_runs.json (runs, roles,
paired comparisons, sections, the bundle chapter, input files); the prose is in text banks (chapters/*.txt) and chapter files (chapters/*.md); no number is typed here.

    wsl <gcar python> bev_fusion/results/build/build_report.py [--config bev_fusion/results/build/report_runs.json] [--out DIR]    (from the repository root, nice -n 15 on a shared box)
    ... --standin t18_fix --compare <compare output>      a TEST build: that run plays the bundle's new run (until the real one exists); the page says so

Writes into the out folder: report.html (+ artifact_page.html, the publish format), report.txt (proofreading text), tables/, videos/, README.txt, mail.txt, check_numbers.txt.
All paths of the config are relative to the problem folder (bev_fusion/); a file that the config names and that does not exist stops the build."""
import argparse
import csv
import glob
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROBLEM = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import bundle_numbers as BN  # noqa: E402
from report_lib import (bar_chart, esc, fail, fill, histogram, html_to_text, img, md_bullets, md_inline, md_tables, need, paired_table, quantiles, read_bank, read_frames,  # noqa: E402
                        read_metrics, read_paired, read_table, render_md, rows_table, table_html)

TITLES = {"summary": ("s1", "1 Summary"), "tried": ("s2", "2 What has been tried"), "system": ("s3", "3 The system"), "results": ("s4", "4 Results on the validation set"),
          "lane": ("s5", "5 Lane markings"), "soa": ("s6", "6 Against the state of the art"), "qual": ("s7", "7 Pictures and videos"), "cost": ("s8", "8 Cost and timing"),
          "weak": ("s9", "9 Weak spots, honest limits, next steps"), "files": ("s10", "10 How to run it, where things are")}     # numbered: the texts refer to these numbers

MAPS = [("drivable area IoU", "seg_fov/seg/dt+0.00/drivable_area/all/all/iou@0", False), ("walkway IoU", "seg_fov/seg/dt+0.00/walkway/all/all/iou@0", False),
        ("car park IoU", "seg_fov/seg/dt+0.00/carpark_area/all/all/iou@0", False), ("pedestrian crossing IoU", "seg_fov/seg/dt+0.00/ped_crossing/all/all/iou@0", False),
        ("lane divider IoU (0.5 m)", "seg_fov/seg/dt+0.00/lane_divider/all/all/iou@0.5", False), ("road divider IoU (1 m)", "seg_fov/seg/dt+0.00/road_divider/all/all/iou@1", False),
        ("drivable IoU at dt -0.5 s", "seg_fov/seg/dt-0.50/drivable_area/all/all/iou@0", False), ("drivable IoU at dt +0.5 s", "seg_fov/seg/dt+0.50/drivable_area/all/all/iou@0", False),
        ("drivable ROC-AUC", "curve/dt+0.00/drivable_area/roc_auc", False), ("walkway AP", "curve/dt+0.00/walkway/ap", False), ("lane divider AP", "curve/dt+0.00/lane_divider/ap", False)]
OBJ = [("mAP (fusion area)", "detf/dt+0.00/mean/all/mAP", False), ("vehicle AP@2 m", "detf/dt+0.00/vehicle/all/ap@2", False), ("pedestrian AP@2 m", "detf/dt+0.00/pedestrian/all/ap@2", False),
       ("two-wheeler AP@2 m", "detf/dt+0.00/two_wheeler/all/ap@2", False), ("vehicle position error ate (m)", "detf/dt+0.00/vehicle/all/ate", True),
       ("vehicle footprint IoU (dense output)", "seg_obj/seg/dt+0.00/vehicle/all/all/iou@0", False)]
VEL = [("velocity error at moving box centres (m/s)", "vel/dt+0.00/centre/field/moving/epe", True), ("same, zero predictor", "vel/dt+0.00/centre/zero/moving/epe", True),
       ("error on moving vehicle cells (m/s)", "vel/dt+0.00/cells/vehicle_moving/epe", True), ("error on moving pedestrian cells (m/s)", "vel/dt+0.00/cells/pedestrian_moving/epe", True),
       ("error on static cells (m/s)", "vel/dt+0.00/cells/static/epe", True)]
COV = [(f"{cls} recall@2 m, {lab}", f"cov/dt+0.00/{cls}/{k}/recall2m", False) for cls in ("vehicle", "pedestrian") for lab, k in (("camera + radar", "cam_radar"), ("camera only", "cam_only"),
                                                                                                                                  ("radar only", "radar_only"), ("neither sensor", "neither"))]
GEN = [("drivable area IoU", "seg_fov/seg/dt+0.00/drivable_area/all/all/iou@0"), ("walkway IoU", "seg_fov/seg/dt+0.00/walkway/all/all/iou@0"),
       ("car park IoU", "seg_fov/seg/dt+0.00/carpark_area/all/all/iou@0"), ("pedestrian crossing IoU", "seg_fov/seg/dt+0.00/ped_crossing/all/all/iou@0"),
       ("lane divider IoU (0.5 m)", "seg_fov/seg/dt+0.00/lane_divider/all/all/iou@0.5"), ("road divider IoU (1 m)", "seg_fov/seg/dt+0.00/road_divider/all/all/iou@1"),
       ("vehicle AP@2 (whole area)", "det/dt+0.00/vehicle/all/ap@2")]
SIX = GEN[:6]                                                      # the six map layers: the "gap" of the summary is their mean train-minus-validation IoU
LANE = [("lane divider IoU, plain (0 m)", "seg_fov/seg/dt+0.00/lane_divider/all/all/iou@0"), ("lane divider IoU, 0.5 m tolerance", "seg_fov/seg/dt+0.00/lane_divider/all/all/iou@0.5"),
        ("lane divider IoU, 1 m tolerance", "seg_fov/seg/dt+0.00/lane_divider/all/all/iou@1"), ("lane divider zero-crossing F1 (0.5 m)", "seg_fov/seg/dt+0.00/lane_divider:zc/all/all/f1@0.5"),
        ("lane divider distance-field MAE (m, lower is better)", "seg_fov/seg/dt+0.00/lane_divider:zc/all/all/field_mae"), ("lane divider AP (probability)", "curve/dt+0.00/lane_divider/ap"),
        ("lane divider ROC-AUC", "curve/dt+0.00/lane_divider/roc_auc"), ("lane divider calibration error (ECE)", "curve/dt+0.00/lane_divider/ece"),
        ("lane divider IoU 0.5 m, 0 - 20 m", "seg_fov/seg/dt+0.00/lane_divider/all/0-20/iou@0.5"), ("lane divider IoU 0.5 m, 20 - 40 m", "seg_fov/seg/dt+0.00/lane_divider/all/20-40/iou@0.5"),
        ("lane divider IoU 0.5 m, 40 - 60 m", "seg_fov/seg/dt+0.00/lane_divider/all/40-60/iou@0.5"), ("lane divider IoU 0.5 m, predicted 0.5 s ahead (dt +0.5)", "seg_fov/seg/dt+0.50/lane_divider/all/all/iou@0.5"),
        ("lane divider IoU 0.5 m, 0.5 s back (dt -0.5)", "seg_fov/seg/dt-0.50/lane_divider/all/all/iou@0.5"),
        ("road divider IoU, 1 m tolerance", "seg_fov/seg/dt+0.00/road_divider/all/all/iou@1"), ("road divider IoU, 0.5 m tolerance", "seg_fov/seg/dt+0.00/road_divider/all/all/iou@0.5"),
        ("road divider zero-crossing F1 (0.5 m)", "seg_fov/seg/dt+0.00/road_divider:zc/all/all/f1@0.5"), ("road divider AP", "curve/dt+0.00/road_divider/ap"),
        ("road divider ROC-AUC", "curve/dt+0.00/road_divider/roc_auc"), ("road divider IoU 1 m, 0 - 20 m", "seg_fov/seg/dt+0.00/road_divider/all/0-20/iou@1"),
        ("road divider IoU 1 m, 20 - 40 m", "seg_fov/seg/dt+0.00/road_divider/all/20-40/iou@1"), ("road divider IoU 1 m, 40 - 60 m", "seg_fov/seg/dt+0.00/road_divider/all/40-60/iou@1"),
        ("pedestrian crossing IoU (plain)", "seg_fov/seg/dt+0.00/ped_crossing/all/all/iou@0"), ("pedestrian crossing AP", "curve/dt+0.00/ped_crossing/ap"),
        ("pedestrian crossing ROC-AUC", "curve/dt+0.00/ped_crossing/roc_auc")]
SOA_LAYERS = [("drivable area", "drivable_area", "iou@0"), ("walkway", "walkway", "iou@0"), ("car park", "carpark_area", "iou@0"), ("pedestrian crossing", "ped_crossing", "iou@0"),
              ("lane divider (0.5 m tolerance)", "lane_divider", "iou@0.5"), ("road divider (0.5 m tolerance)", "road_divider", "iou@0.5")]
VIDEO_GLOBS = ("{eval}/showcase/qualitative/videos/*_overview.mp4", "{eval}/suite/qualitative/videos/*_dashboard.mp4", "{eval}/suite/qualitative/videos/*_overview.mp4")
ZC = (("p", "p@0.5"), ("r", "r@0.5"), ("f1", "f1@0.5"))


# ------------------------------------------------------------------------------------------------ what the sections share
class Ctx:
    """the config, the runs that have an evaluation, their tables, the text banks and the facts (numbers the texts ask for by name: @@name@@, computed when first asked)"""

    def __init__(self, cfg, out):
        self.cfg, self.out, self.n = cfg, out, cfg.get("frames", 1000)
        self.bank = {}
        for p in cfg.get("text", []):
            self.bank.update(read_bank(p))
        def has_table(r, what):
            """True when the run has its evaluation table; a run that may be missing ("optional") is skipped with a note, any other missing run stops the build"""
            if os.path.isfile(f"{r['eval']}/by_n/table_n{self.n}.txt"):
                return True
            if r.get("optional"):
                print(f"skipped (no evaluation yet): {r['id']}")
                return False
            fail(f"{what} {r['id']}: no evaluation table {r['eval']}/by_n/table_n{self.n}.txt (a run that may be missing says \"optional\": true in the config)")

        self.runs = []
        for r in cfg["runs"]:
            r = dict(r, role=r.get("role", "arm"))
            r.setdefault("eval", f"runs/{r['id']}/eval")
            if has_table(r, "run"):
                self.runs.append(r)
        self.extra = []         # runs of the experiment chapters (bundle2, ...): their tables are read, but they do not enter the general tables of the other sections
        for r in cfg.get("extra_runs", []):
            r = dict(r, role="arm")
            r.setdefault("eval", f"runs/{r['id']}/eval")
            if has_table(r, "extra run"):
                self.extra.append(r)
        self.by_id = {r["id"]: r for r in self.runs + self.extra}
        self.T = {r["id"]: read_table(f"{r['eval']}/by_n/table_n{self.n}.txt") for r in self.runs + self.extra}
        bases = [r for r in self.runs if r["role"] == "baseline"]
        if len(bases) > 1:
            fail("more than one run has the role baseline")
        self.base = bases[0] if bases else None
        self._memo = {}

    def text(self, name, optional=False):
        if name not in self.bank:
            if optional:
                return ""
            fail(f"no text block '{name}' in the text banks {self.cfg.get('text')}")
        return fill(self.bank[name], self.fact, f"text block '{name}'")

    def memo(self, name, fn):
        if name not in self._memo:
            self._memo[name] = fn()
        return self._memo[name]

    def baseline(self):
        if self.base is None:
            fail("no run has the role baseline")
        return self.base

    def paths(self, name):
        """the input file(s) that the config names under inputs.<name> (a path or a list of paths); every one must exist"""
        p = self.cfg.get("inputs", {}).get(name)
        if not p:
            fail(f"report_runs.json: inputs.{name} is not set")
        return [need(x, f"input '{name}'") for x in (p if isinstance(p, list) else [p])]

    def path(self, name):
        return self.paths(name)[0]

    def eval_file(self, run, rel):
        return need(f"{run['eval']}/{rel}", f"run {run['id']}")

    def train_table(self, run):
        for p in (f"{run['eval']}/by_n/table_train_n{self.n}.txt", f"{run['eval']}/table_train.txt"):
            if os.path.isfile(p):
                return read_table(p)
        return {}

    # ---- facts
    def fact(self, name):
        if name.startswith("ms_"):
            return self.ms(name[3:]) if name[3:] in self.by_id else None
        f = getattr(self, "f_" + name, None)
        return f() if f else None

    def ms(self, run_id):
        d = json.load(open(self.eval_file(self.by_id[run_id], "timing_modules.json"), encoding="utf-8"))
        return f"{d['inference_ms']['total_ms']:.1f}"

    def metrics(self, run):
        """all numbers of the run's metrics_n<N>.json (more keys than the table: plain IoU of the lines, range bins ...)"""
        return self.memo("metrics " + run["id"], lambda: read_metrics(self.eval_file(run, f"by_n/metrics_n{self.n}.json")))

    def val(self, key, run=None, fmt="{:.3f}"):
        """a number of the baseline (or `run`) from its table, else from its metrics file; a run without it stops the build"""
        run = run or self.baseline()
        v = self.T[run["id"]].get(key, self.metrics(run).get(key) if os.path.isfile(f"{run['eval']}/by_n/metrics_n{self.n}.json") else None)
        if v is None:
            fail(f"run {run['id']} has no key {key}")
        return fmt.format(v)

    def lane_stats(self, run):
        """lane-divider IoU over the validation frames that have at least 30 ground-truth lane-divider cells in view"""
        vals = []
        for lab, s, sup in read_frames(self.eval_file(run, f"suite/quant/eval_curves_fusion_vel_n{self.n}/frames.jsonl")):
            v, n = s.get("seg.lane_divider_iou"), sup.get("seg.lane_divider_iou") or 0
            if v is not None and n >= 30:
                vals.append(v)
        return vals

    def f_head(self):
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() or "unknown"

    def f_frames(self):
        return self.n

    def f_new_label(self):
        return esc(bundle(self)["new_label"])

    def f_ctl_label(self):
        return esc(bundle(self)["ctl_label"])

    def f_done(self):
        return esc(", ".join(r["label"] for r in self.runs))

    def f_key_table(self):
        return summary_table(self)

    def baseline_lane(self):
        return self.memo("lane", lambda: self.lane_stats(self.baseline()))

    def f_lane_n(self):
        return len(self.baseline_lane())

    def f_lane_median(self):
        return f"{quantiles(self.baseline_lane())[2]:.2f}"

    def f_lane_lo(self):
        return f"{100 * sum(x < 0.1 for x in self.baseline_lane()) / len(self.baseline_lane()):.0f}"

    def f_lane_hi(self):
        return f"{100 * sum(x >= 0.5 for x in self.baseline_lane()) / len(self.baseline_lane()):.0f}"

    def zc(self, k):
        return self.val("seg_fov/seg/dt+0.00/lane_divider:zc/all/all/" + dict(ZC)[k], fmt="{:.2f}")

    def f_zc_p(self):
        return self.zc("p")

    def f_zc_r(self):
        return self.zc("r")

    def f_zc_f1(self):
        return self.zc("f1")

    def f_lane_plain(self):
        return self.val("seg_fov/seg/dt+0.00/lane_divider/all/all/iou@0")

    def f_vel_field(self):
        return self.val("vel/dt+0.00/centre/field/moving/epe", fmt="{:.2f}")

    def f_vel_zero(self):
        return self.val("vel/dt+0.00/centre/zero/moving/epe", fmt="{:.2f}")

    def f_vel_zero3(self):
        return self.val("vel/dt+0.00/centre/zero/moving/epe")

    def f_gap6(self):
        base, tr = self.T[self.baseline()["id"]], self.train_table(self.baseline())
        if not tr:
            fail(f"run {self.baseline()['id']} has no train table (the gap of the summary)")
        return f"{sum(tr[k] - base[k] for _, k in SIX) / len(SIX):.3f}"

    def f_nvid(self):
        return sum(len(names) for _, _, names in videos(self))

    def f_spread(self):
        return f"{timing_spread(self):.1f}"

    def f_architecture_svg(self):
        return open(self.path("architecture_svg"), encoding="utf-8").read()

    # the state-of-the-art re-scorings (maps against GaussianCaR, front sector against DualBEV)
    def soa(self):
        def read():
            t = md_tables(open(self.path("soa_maps"), encoding="utf-8").read())[0][2]               # the first file is the baseline's
            fr = md_tables(open(self.path("soa_front"), encoding="utf-8").read())[0][2]
            return {(r[0], r[1]): (r[4], r[5], r[6]) for r in t if len(r) >= 7}, fr
        return self.memo("soa", read)

    def f_soa_parts(self):
        maps = self.soa()[0]
        return "; ".join(f"{n} {maps[(l, m)][0]} / {maps[(l, m)][1]} ({int(round(float(maps[(l, m)][2]) * 100))} %)" for n, l, m in SOA_LAYERS if (l, m) in maps)

    def f_front_dual(self):
        return self.soa()[1][0][1]

    def f_front_ours(self):
        return self.soa()[1][1][1]

    def f_front_diff(self):
        return self.soa()[1][1][2]

    def f_maps_table(self):
        maps = self.soa()[0]
        rows = [[name, *maps[(layer, metric)][:2], maps[(layer, metric)][2], self.cfg["soa"]["old_shares"][layer]] for name, layer, metric in SOA_LAYERS]
        return table_html(["layer", "ours (baseline)", "GaussianCaR", "ours as a share", "share of the dual chain after 3000 steps in the older scoring"], rows, left=(0,))

    def f_carpark_soa2(self):
        return f"{float(self.soa()[0][('carpark_area', 'iou@0')][0]):.2f}"

    def f_carpark_val2(self):
        return self.val("seg_fov/seg/dt+0.00/carpark_area/all/all/iou@0", fmt="{:.2f}")

    def f_drivable_soa2(self):
        return f"{float(self.soa()[0][('drivable_area', 'iou@0')][0]):.2f}"

    def f_drivable_val2(self):
        return self.val("seg_fov/seg/dt+0.00/drivable_area/all/all/iou@0", fmt="{:.2f}")

    def f_lane_soa3(self):
        return self.soa()[0][("lane_divider", "iou@0.5")][0]

    def f_lane_val3(self):
        return self.val("seg_fov/seg/dt+0.00/lane_divider/all/all/iou@0.5")

    def f_drivable_soa3(self):
        return self.soa()[0][("drivable_area", "iou@0")][0]

    def f_drivable_val3(self):
        return self.val("seg_fov/seg/dt+0.00/drivable_area/all/all/iou@0")

    def f_soa_sentence(self):
        return self.text("soa_sentence")


def timing(ctx, run):
    return json.load(open(ctx.eval_file(run, "timing_modules.json"), encoding="utf-8"))


def timing_spread(ctx):
    totals = [timing(ctx, r)["inference_ms"]["total_ms"] for r in ctx.runs]
    return max(totals) - min(totals)


def summary_table(ctx):
    """the numbers of the summary, from the tables of the runs. cfg.summary.table = "bundle": the bundle rows (new run against the control, as in the chapter); else the rows
    of cfg.summary.rows = [name, key | [keys] | "@@fact@@", reading] for the baseline"""
    s = ctx.cfg.get("summary", {})
    if s.get("table") == "bundle":
        b = bundle(ctx)
        return BN.table_key(b["rows"], b["new_label"], b["ctl_label"])
    rows = []
    for name, spec, reading in s["rows"]:
        if isinstance(spec, list):
            cell = " / ".join(ctx.val(k) for k in spec)
        elif spec.startswith("@@"):
            cell = fill(spec, ctx.fact, "summary row")
        else:
            cell = ctx.val(spec)
        rows.append([name, cell, fill(reading, ctx.fact, "summary row")])
    return table_html(["quantity", "baseline", "reading"], rows, left=(0, 2))


# ------------------------------------------------------------------------------------------------ the bundle chapter
def side(ctx, r):
    """what the bundle table needs of a run: its table and its idle-GPU ms per frame (None when there is no timing file)"""
    t = timing(ctx, r) if os.path.isfile(f"{r['eval']}/timing_modules.json") else None
    return dict(id=r["id"], table=ctx.T[r["id"]], ms=t["inference_ms"]["total_ms"] if t else None, idle=bool(t and t.get("idle")))


def bundle(ctx, key="bundle"):
    """the rows of the bundle table (bundle_numbers.collect), once; key = the config entry ("bundle" = the first chapter, "bundle2", ... further chapters)"""
    def build():
        b = ctx.cfg.get(key)
        if not b:
            fail(f"report_runs.json has no \"{key}\" (the new run and its control)")
        new, ctl = ctx.by_id.get(b["run"]), ctx.by_id.get(b["control"])
        if new is None or ctl is None:
            fail(f"{key}: run {b['run']} or control {b['control']} is not among the runs with an evaluation")
        if not os.path.isfile(b["compare"]):
            fail(f"{key}: {b['compare']} does not exist; make it (CPU, from bev_fusion/): python evaluation/bev_suite.py compare runs/{b['control']} runs/{b['run']} "
                 f"--curves --fusion --velocity --n {ctx.n} --boot 1000 > {b['compare']}")
        head, paired = read_paired(b["compare"])
        rows = BN.collect(b["rows"], side(ctx, new), side(ctx, ctl), paired, head, test=bool(ctx.cfg.get("test")))
        return dict(rows=rows, new=new, ctl=ctl, new_label=b.get("new_label", new["label"]), ctl_label=b.get("control_label", ctl["label"]), band=b.get("band", 0.010), head=head,
                    weak_below=b.get("weak_below", 0.5))
    return ctx.memo(key, build)


def bundle_tables(ctx, key="bundle"):
    """(the numbers table, the weak-spot table) of the bundle chapter, as html; the weak-spot table is empty when the entry says `"weak_table": false`"""
    b = bundle(ctx, key)
    gaps = {}
    tr, val = ctx.train_table(b["new"]), ctx.T[b["new"]["id"]]
    for r in b["rows"]:
        if r["key"] in tr and r["key"] in val:
            gaps[r["key"]] = tr[r["key"]] - val[r["key"]]
    full = (f"<h3>The numbers (VAL, {ctx.n} frames)</h3><p class=small>{esc(b['new_label'])}: <code>{esc(b['new']['id'])}</code>; {esc(b['ctl_label'])}: <code>{esc(b['ctl']['id'])}</code>. "
            f"Maps on the cells the front camera sees, detection on the fusion area, ms per frame at batch 1 on an idle GPU. The difference is new minus control (green = better, red = worse beyond "
            f"{b['band']:g}); the paired column is the scene bootstrap of the same frames (b = new, a = control), the reading follows its interval. `n/a` = no paired test for the metric (the compare output bootstraps the headline numbers of the tasks only); the difference is then the unpaired one.</p>"
            + BN.table_full(b["rows"], b["new_label"], b["ctl_label"], b["band"]) + f"<p class=cap>{esc(b['head'])}</p>")
    weak = (f"<h3>Weak spots (updated from the numbers)</h3><p class=small>The outputs of the table that are scores between 0 and 1, lowest of the new run first; flagged when below {b['weak_below']:g}. "
            "The gap is the new run's score on its own training scenes minus the validation score.</p>"
            + BN.table_weak(b["rows"], gaps, b["weak_below"], b["new_label"], b["ctl_label"]))
    if ctx.cfg[key].get("weak_table") is False:
        weak = ""
    return full, weak


def chapter_html(text, full, weak):
    """the chapter file (markdown) as html; a line `@@TABLE@@` / `@@WEAK@@` puts the numbers / the weak-spot table there; without the markers both come first (numbers, weak spots, text)"""
    marker = lambda name: re.search(rf"^@@{name}@@[ \t]*$", text, flags=re.M)         # noqa: E731  a marker is a line of its own
    if not marker("TABLE"):
        text = "@@TABLE@@\n" + text
    if not marker("WEAK"):
        text = re.sub(r"^@@TABLE@@[ \t]*$", "@@TABLE@@\n@@WEAK@@", text, count=1, flags=re.M)
    parts = re.split(r"^@@(TABLE|WEAK)@@[ \t]*$", text, flags=re.M)
    return "".join(render_md(p, bullets=True) if i % 2 == 0 else {"TABLE": full, "WEAK": weak}[p] for i, p in enumerate(parts))      # odd pieces: the captured marker names


def sec_bundle(ctx, key="bundle"):
    return chapter_html(open(need(ctx.cfg[key]["chapter"], f"{key} chapter"), encoding="utf-8").read(), *bundle_tables(ctx, key))


# ------------------------------------------------------------------------------------------------ sections
def sec_summary(ctx):
    return ctx.text("summary")


def sec_tried(ctx):
    text = open(ctx.path("history_digest"), encoding="utf-8").read()
    exp = next((t for t in md_tables(text) if t[0].startswith("1. Table")), None)
    if exp is None:
        fail("history digest: no table '1. Table ...'")
    out = [ctx.text("tried_intro")]
    for row in exp[2]:
        if len(row) < 7:
            continue
        ident, question, changed, numbers, verdict, lesson, source = row[:7]
        out.append(f"<div class=exp><h4>{md_inline(ident)}</h4><p><span class=k>Question.</span> {md_inline(question)}</p><p><span class=k>What was changed.</span> {md_inline(changed)}</p>"
                   f"<p><span class=k>Decisive numbers.</span> {md_inline(numbers)}</p><p><span class=k>Verdict.</span> {md_inline(verdict)}</p><p><span class=k>Lesson.</span> {md_inline(lesson)}</p>"
                   f"<p class=small>source: {md_inline(source)}</p></div>")
    for head, a, b in (("What worked", "### (a)", "### (b)"), ("What did not work or stayed flat (with the reason the cards give)", "### (b)", "### (c)"),
                       ("Open questions the cards name", "### (c)", "## 3.")):
        items = md_bullets(text, a, b)
        if not items:
            fail(f"history digest: no bullets between '{a}' and '{b}'")
        out.append(f"<h3>{head}</h3><ul>" + "".join(f"<li>{md_inline(i)}</li>" for i in items) + "</ul>")
    caveats = md_bullets(text, "## 3. Caveats", "## 4.")
    if not caveats:
        fail("history digest: no bullets in '## 3. Caveats'")
    return "".join(out) + "<h3>Caveats of the digest</h3><ul>" + "".join(f"<li>{md_inline(i)}</li>" for i in caveats) + "</ul>"


def sec_system(ctx):
    return ctx.text("system")


def sec_results(ctx):
    labels, tabs = [r["label"] for r in ctx.runs], [ctx.T[r["id"]] for r in ctx.runs]
    out = [ctx.text("results_intro"), ctx.text("results_warn", optional=True)]
    out.append("<h3>Maps</h3>" + rows_table(MAPS, tabs, labels))
    out.append("<h3>Objects</h3>" + rows_table(OBJ, tabs, labels))
    out.append("<h3>Velocity (lower is better)</h3>" + rows_table(VEL, tabs, labels))
    out.append("<h3>Detection recall by the sensors that see the object (does the radar help?)</h3>" + ctx.text("results_cov_note") + rows_table(COV, tabs, labels))
    if ctx.cfg.get("compare"):
        out.append("<h3>Paired comparisons (the same frames, scenes resampled 1000 times)</h3>")
        for c in ctx.cfg["compare"]:
            out.append(f"<h4>{esc(c['title'])}</h4>" + paired_table(c["file"], c["a"], c["b"]))
    gen = [(r["label"], ctx.T[r["id"]], ctx.train_table(r)) for r in ctx.runs if ctx.train_table(r)]
    if gen:
        rows = []
        for n, k in GEN:
            if not all(k in tr and k in v for _, v, tr in gen):
                continue
            rows.append([n] + [f"{tr.get(k, float('nan')):.3f} / {v.get(k, float('nan')):.3f} / {tr.get(k, float('nan')) - v.get(k, float('nan')):+.3f}" for _, v, tr in gen])
        out.append("<h3>Train against validation (the generalisation gap)</h3>" + ctx.text("results_gen_note") + table_html(["metric"] + [l for l, _, _ in gen], rows, left=(0,)))
    return "".join(out)


def sec_lane(ctx):
    out = [ctx.text("lane_51")]
    M = {r["id"]: ctx.metrics(r) for r in ctx.runs}
    rows = []
    for n, k in LANE:
        cells, base = [], None
        for r in ctx.runs:
            v = ctx.T[r["id"]].get(k, M[r["id"]].get(k))
            if v is None:
                cells.append("-")
            elif base is None:
                base = v
                cells.append(f"{v:.3f}")
            else:
                cells.append(f"{v:.3f} <span class=delta>{v - base:+.3f}</span>")
        rows.append([n] + cells)
    out.append(f"<h3>5.2 The numbers (validation, {ctx.n} frames)</h3>" + table_html(["metric"] + [r["label"] for r in ctx.runs], rows, left=(0,)) + ctx.text("lane_52_note"))
    out.append("<h3>5.3 How the lane divider quality is distributed over frames</h3>")
    for rid in ctx.cfg["lane"]["histograms"]:
        r = ctx.by_id[rid]
        ls = ctx.lane_stats(r)
        if ls:
            q = quantiles(ls)
            out.append(histogram(ls, f"{r['label']}: lane-divider IoU (0.5 m) per validation frame, {len(ls)} frames with at least 30 true divider cells",
                                 note=f"quantiles 10/25/50/75/90 %: {q[0]:.2f} / {q[1]:.2f} / {q[2]:.2f} / {q[3]:.2f} / {q[4]:.2f}; mean {sum(ls) / len(ls):.2f}"))
    out.append(ctx.text("lane_53"))
    out.append("<h3>5.4 What moved the lane divider (the ladder of experiments)</h3>" + ladder(ctx))
    out.append(lane_gallery(ctx))
    out.append(ctx.text("lane_reading"))
    out.append("<h3>5.6 Against the state of the art</h3>" + lane_soa(ctx))
    return "".join(out)


def ladder(ctx):
    """the bars of the cards' values over the experiments (cfg.ladder: a literal value copied from a card, or the value of a run's table)"""
    charts = []
    for c in ctx.cfg["ladder"]:
        items = [(i["label"], i["value"] if "value" in i else ctx.T[i["run"]][i["key"]], i["gt"]) for i in c["items"]]
        charts.append(bar_chart(items, c["title"], vmax=c["vmax"], note=ctx.text("ladder_note")))
    return "".join(charts)


def lane_gallery(ctx):
    base = ctx.baseline()
    d = f"{base['eval']}/showcase/qualitative"
    picks = []
    for ln in open(ctx.path("lane_picks"), encoding="utf-8"):
        m = re.match(r"PICK\s+(\w+)\s+(scene-\d+)/(\d+)\s+lane_iou\s+([0-9.]+)", ln)
        if m:
            picks.append(m.groups())
    out = [ctx.text("lane_gallery_intro")]
    found = 0
    for kind in ("worst", "median", "best"):
        imgs = []
        for k, scene, idx, iou in picks:
            if k != kind:
                continue
            cand = glob.glob(f"{d}/**/{scene}*{int(idx):03d}*overview*.png", recursive=True) + glob.glob(f"{d}/**/{scene}_{int(idx)}*overview*.png", recursive=True) + glob.glob(f"{d}/**/*{scene}*{int(idx)}*.png", recursive=True)
            if cand:
                found += 1
                imgs.append(img(cand[0], 1000) + f"<div class=cap>{kind}: {scene} frame {int(idx)}, lane-divider IoU {iou} (the frame number counts from the start of the scene)</div>")
        if imgs:
            out.append(f"<h4>{kind.capitalize()} frames</h4>" + "".join(imgs))
    if not found:
        fail(f"the lane gallery: no picture of the frames named in {ctx.path('lane_picks')} under {d}")
    return "".join(out)


def compact_maps(path, layers):
    """the rows of the first table (threshold 0.5, all ranges) of a soa_maps_v8 result file for some layers, as html"""
    text = open(need(path, "soa maps"), encoding="utf-8").read()
    tabs = md_tables(text)
    hd, rows = tabs[0][1], [r for r in tabs[0][2] if r[0] in layers and len(r) >= 7]
    head = text.split("\n\n")[1] if "\n\n" in text else ""
    return (f"<p class=small>{md_inline(head)}</p><table class=wide><tr>" + "".join(f"<th class=l>{esc(c)}</th>" for c in hd) + "</tr>" +
            "".join("<tr>" + "".join(f"<td>{md_inline(c)}</td>" for c in r) + "</tr>" for r in rows) + "</table>" +
            "<p class=small>All tables (best threshold, by range, the distance-field numbers, the checks) are in section 6.3.</p>")


def lane_soa(ctx):
    pub = ctx.cfg["lane_published"]
    s = "<table class=wide><tr>" + "".join(f"<th class=l>{c}</th>" for c in pub["header"]) + "</tr>"
    for r in pub["rows"]:
        s += "<tr>" + "".join(f"<td class=l>{esc(c)}</td>" for c in r) + "</tr>"
    ours = (f"{ctx.val('seg_fov/seg/dt+0.00/lane_divider/all/all/iou@0.5')} at 0.5 m tolerance; {ctx.val('seg_fov/seg/dt+0.00/lane_divider/all/all/iou@0')} plain; "
            f"{ctx.val('seg_fov/seg/dt+0.00/lane_divider/all/all/iou@1')} at 1 m")
    s += "<tr>" + "".join(f"<td class=l>{esc(c)}</td>" for c in (pub["ours"][0], ours, pub["ours"][1])) + "</tr>"
    s += "</table>" + ctx.text("lane_published_note")
    s += "<h4>Same frames, same ground truth, same metric (our re-scoring of GaussianCaR's cached logits)</h4>"
    return s + compact_maps(soa_files(ctx)[0], ("lane_divider", "road_divider", "ped_crossing"))


def soa_files(ctx):
    return ctx.paths("soa_maps") + ctx.paths("soa_front")


def render_result_md(path):
    return render_md(open(need(path, "re-scoring result"), encoding="utf-8").read())


def sec_soa(ctx):
    text = open(ctx.path("soa_digest"), encoding="utf-8").read()
    byhead = {h: (hd, rows) for h, hd, rows in md_tables(text)}

    def digest_table(key):
        if key not in byhead:
            fail(f"soa digest: no table '{key}'")
        return byhead[key]

    def wide(hd, rows):
        return ("<table class=wide><tr>" + "".join(f"<th class=l>{esc(c)}</th>" for c in hd) + "</tr>" +
                "".join("<tr>" + "".join(f"<td>{md_inline(c)}</td>" for c in r) + "</tr>" for r in rows) + "</table>")

    out = [ctx.text("soa_howto"), "<h3>6.1 Published state-of-the-art numbers (as published; protocol in the last column)</h3>"]
    for key, title in (("1A BEV map segmentation (nuScenes val; IoU in %)", "BEV map segmentation (IoU in per cent)"),
                       ("1B 3D detection (nuScenes val unless stated; NDS and mAP in %, errors dimensionless)", "3D detection")):
        hd, rows = digest_table(key)
        out.append(f"<details><summary>{esc(title)}: {len(rows)} rows</summary>{wide(hd, rows)}</details>")
    out.append(ctx.text("soa_61_note"))
    hd, rows = digest_table("2 Stored re-scorings of SOA models with OUR metric on OUR frames")
    out.append("<h3>6.2 The repository's own re-scorings of state-of-the-art models (same frames, our metric, older ground truth)</h3>")
    for r in rows:
        if len(r) < 10:
            out.append(f"<div class=exp><h4>{md_inline(r[0])}: {md_inline(r[1][:160]) if len(r) > 1 else ''}</h4><p>{md_inline(' / '.join(c for c in r[2:] if c and c != '-'))}</p></div>")
        if len(r) >= 10:
            out.append(f"<div class=exp><h4>{md_inline(r[0])}: {md_inline(r[1][:160])}</h4><p><span class=k>SOA model.</span> {md_inline(r[2])}</p><p><span class=k>Ours.</span> {md_inline(r[3])}</p>"
                       f"<p><span class=k>Ground truth, frames, metric.</span> {md_inline(r[4])}; {md_inline(r[5])}; {md_inline(r[6])}</p><p><span class=k>Numbers.</span> {md_inline(r[7])}</p>"
                       f"<p><span class=k>Statement of the card.</span> {md_inline(r[8])}</p><p class=small>{md_inline(r[9])}</p></div>")
    out.append("<h3>6.3 New: the like-for-like re-scoring on the current ground truth (v8)</h3>")
    out.append(ctx.text("soa_reading"))
    for p in soa_files(ctx):
        out.append(render_result_md(p))
    out.append("<h3>6.4 Our numbers next to the published ones, and why they cannot be ranked</h3>")
    for key in ("3A Maps", "3B Detection and velocity"):
        hd, rows = digest_table(key)
        out.append(f"<h4>{esc(key)}</h4>{wide(hd, rows)}")
    sec4 = re.search(r"## 4 How to read this\n\n(.+?)\n\n## 5", text, re.S)
    if not sec4:
        fail("soa digest: no section '## 4 How to read this'")
    out.append(f"<div class=card><b>What a fair claim is today.</b> {md_inline(sec4.group(1))}</div>")
    return "".join(out)


def videos(ctx):
    """[(run, label-or-long text, [file names])]: the videos of every run are copied next to the page (out/videos/<run>/), once"""
    def build():
        groups = []
        for r in ctx.runs:
            pats = r.get("videos") or [p.format(eval=r["eval"]) for p in VIDEO_GLOBS]
            files = [fn for pat in pats for fn in sorted(glob.glob(pat))]
            if files:
                dst = f"{ctx.out}/videos/{r['id']}"
                os.makedirs(dst, exist_ok=True)
                names = []
                for fn in files:
                    to = os.path.join(dst, os.path.basename(fn))
                    if not os.path.exists(to) or os.path.getsize(to) != os.path.getsize(fn):
                        shutil.copyfile(fn, to)
                    names.append(os.path.basename(fn))
                groups.append((r["id"], r.get("long", r["label"]), names))
        return groups
    return ctx.memo("videos", build)


def sec_qual(ctx):
    out = [ctx.text("qual_intro")]
    r = ctx.baseline()
    pics = sorted(glob.glob(f"{r['eval']}/images/*_overview.png"))[:6]
    for fn in pics:
        out.append(img(fn, 1000) + f"<div class=cap>{esc(r['label'])}: {esc(os.path.basename(fn))}: front camera with the true boxes and lines, true BEV, prediction, error maps of drivable area, vehicle footprint and lane dividers</div>")
    for k in (0, 2):
        fn = f"{r['eval']}/modules/frame_{k}.png"
        if os.path.exists(fn):
            out.append(img(fn, 1100) + f"<div class=cap>{esc(r['label'])}: the module probe: what every stage of the chain produces for one validation frame</div>")
    out.append(ctx.text("videos_intro"))
    for rid, long, names in videos(ctx):
        out.append(f"<h3>{esc(long)}</h3><div class=vgrid>")
        for n in names:
            out.append(f"<figure><video controls preload=none src='videos/{rid}/{n}'></video><figcaption class=small>{esc(n)}</figcaption></figure>")
        out.append("</div>")
    if not videos(ctx):
        out.append("<p><em>no videos yet</em></p>")
    return "".join(out)


def ledger_hours(path):
    out = {}
    for row in csv.reader(open(need(path, "ledger"), encoding="utf-8")):
        if row and len(row) > 5 and row[3] == "ok":
            out[row[0]] = (row[1], row[2], float(row[5]) / 3600 if row[5] else 0.0)
    return out


def sec_cost(ctx):
    hrs = ledger_hours(ctx.cfg.get("ledger", "records/LEDGER.csv"))
    rows = []
    for rid, label in ctx.cfg["cost_rows"]:
        if rid not in hrs:
            fail(f"cost_rows: the ledger has no finished row '{rid}'")
        rows.append([rid, esc(label), hrs[rid][0].replace("T", " "), f"{hrs[rid][2]:.1f} h"])
    trows, per = [], None
    for r in ctx.runs:
        d = timing(ctx, r)
        inf, tm = d["inference_ms"], d.get("train_microbatch_ms") or {}
        trows.append([esc(r["label"]), f"{inf['total_ms']:.1f}", f"{sum(inf['per_frame_ms'].values()):.1f}", f"{inf['unattributed_ms']:+.1f}",
                      f"{tm.get('forward_loss', float('nan')):.0f} / {tm.get('backward', float('nan')):.0f}", "yes" if d.get("idle") else "NO (do not quote)"])
        per = per or (r["label"], inf)
    out = ctx.text("cost") + ctx.text("cost_timing_intro")
    out += table_html(["run", "total (ms per frame)", "sum of modules (ms)", "unattributed (ms)", "training micro-batch, forward+loss / backward (ms)", "idle"], trows, left=(0,))
    mods = sorted(per[1]["per_frame_ms"].items(), key=lambda kv: -kv[1])
    out += (f"<h4>Where the time goes ({esc(per[0])}, ms per frame)</h4>"
            + table_html(["module", "ms per frame", "calls", "share of the module sum"],
                         [[esc(k), f"{v:.2f}", str(per[1]['calls'][k]), f"{100 * v / sum(per[1]['per_frame_ms'].values()):.1f} %"] for k, v in mods], left=(0,)))
    return out + "<h3>Wall-clock of the large runs</h3>" + table_html(["run", "what", "started", "duration"], rows, left=(0, 1))


def sec_weak(ctx):
    return ctx.text("weak")


def sec_files(ctx):
    return ctx.text("howto") + "<h3>Where things are</h3>" + ctx.text("files")


SECTIONS = {"summary": sec_summary, "tried": sec_tried, "system": sec_system, "results": sec_results, "lane": sec_lane, "soa": sec_soa, "qual": sec_qual, "cost": sec_cost,
            "weak": sec_weak, "files": sec_files}


# ------------------------------------------------------------------------------------------------ assembly
def page(ctx):
    cfg = ctx.cfg
    secs = []
    for name in cfg["sections"]:
        if re.fullmatch(r"bundle\d*", name):        # "bundle" = the first experiment chapter, "bundle2", ... further ones (each with its own config entry)
            secs.append((name, cfg[name]["title"], sec_bundle(ctx, name)))
        elif name in SECTIONS:
            secs.append((TITLES[name][0], TITLES[name][1], SECTIONS[name](ctx)))
        else:
            fail(f"unknown section '{name}' in the config")
    toc = "<nav class=toc>" + "".join(f"<a href='#{i}'>{esc(t)}</a>" for i, t, _ in secs) + "</nav>"
    body = "".join(f"<h2 id='{i}'>{esc(t)}</h2>{b}" for i, t, b in secs)
    banner = f"<div class=warn><b>TEST BUILD.</b> {esc(cfg['test'])}</div>" if cfg.get("test") else ""
    main_html = f"<main><h1>{esc(cfg['title'])}</h1>{banner}<p class=lead>{ctx.text('lead')}</p>{toc}{body}</main>"
    main_html = main_html.replace("<table", "<div class=scroll><table").replace("</table>", "</table></div>")      # a wide table scrolls inside its own box
    css = open(os.path.join(HERE, "page.css"), encoding="utf-8").read()
    head_bits = ('<title>BEV Result Report</title><meta name="claude:viewport-fit-sides" content="cover">'
                 '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">'
                 f"<style>{css}</style>")
    return (f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1,viewport-fit=cover'>{head_bits}</head><body>{main_html}</body></html>",
            head_bits + main_html)


def write_folder(ctx, full, publish):
    out = ctx.out
    os.makedirs(out, exist_ok=True)
    open(f"{out}/report.html", "w", encoding="utf-8").write(full)
    open(f"{out}/artifact_page.html", "w", encoding="utf-8").write(publish)
    open(f"{out}/report.txt", "w", encoding="utf-8").write(html_to_text(full))
    os.makedirs(f"{out}/tables", exist_ok=True)
    for r in ctx.runs:
        for src, name in ((f"{r['eval']}/by_n/table_n{ctx.n}.txt", f"{r['id']}_table_n{ctx.n}.txt"), (f"{r['eval']}/table_train.txt", f"{r['id']}_table_train.txt")):
            if os.path.exists(src):
                shutil.copyfile(src, f"{out}/tables/{name}")
    if "folder_readme" in ctx.bank:
        open(f"{out}/README.txt", "w", encoding="utf-8", newline="\n").write(ctx.bank["folder_readme"] + "\n")


def check_numbers(ctx):
    """the first-pass numbers check (autoresearch/tools/check_numbers.py): the decimal numbers of the page that no source file holds; advisory, the list goes to check_numbers.txt"""
    srcs = [p for pat in ctx.cfg.get("check_sources", []) for p in sorted(glob.glob(pat))]
    if not srcs:
        return
    tool = os.path.join(HERE, "..", "..", "..", "autoresearch", "tools", "check_numbers.py")
    r = subprocess.run([sys.executable, tool, f"{ctx.out}/report.html"] + srcs + ["--show", "120"], capture_output=True, text=True)
    open(f"{ctx.out}/check_numbers.txt", "w", encoding="utf-8").write(r.stdout + r.stderr)
    print((r.stdout.splitlines() or [r.stderr.strip()])[0])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(HERE, "report_runs.json"))
    ap.add_argument("--root", default=PROBLEM, help="the problem folder that the paths of the config are relative to")
    ap.add_argument("--out", help="the out folder (default: the config's, with _test after it for a stand-in build)")
    ap.add_argument("--standin", metavar="RUN", help="TEST build: RUN plays the bundle's new run")
    ap.add_argument("--compare", help="with --standin: the `bev_suite.py compare` output (a = the control, b = RUN)")
    ap.add_argument("--no-mail", action="store_true")
    ap.add_argument("--no-check", action="store_true")
    a = ap.parse_args(argv)
    cfg = json.load(open(os.path.abspath(a.config), encoding="utf-8"))
    out = os.path.abspath(a.out) if a.out else None
    os.chdir(a.root)
    if a.standin:
        old = cfg["bundle"]["run"]
        if a.standin not in [r["id"] for r in cfg["runs"]] or not a.compare:
            fail(f"--standin RUN --compare FILE: RUN must be a run of the config ({a.standin}), and the compare output is needed")
        cfg["runs"] = [r for r in cfg["runs"] if r["id"] != old]                # the new run does not exist yet; the stand-in takes its place everywhere
        cfg = json.loads(json.dumps(cfg).replace(old, a.standin).replace(old.replace("_", ".", 1), a.standin.replace("_", ".", 1)))
        cfg["bundle"]["compare"] = a.compare
        cfg["test"] = f"stand-in: {a.standin} plays the bundle's new run; not a result."
        cfg["out"] = cfg["out"] + "_test"
    ctx = Ctx(cfg, out or cfg["out"])
    full, publish = page(ctx)
    write_folder(ctx, full, publish)
    print("written", f"{ctx.out}/report.html", f"{os.path.getsize(ctx.out + '/report.html') / 1e6:.1f} MB", "runs:", [r["id"] for r in ctx.runs])
    if cfg.get("mail") and not a.no_mail:
        import build_mail
        open(f"{ctx.out}/mail.txt", "w", encoding="utf-8", newline="\n").write(build_mail.build(ctx))
    if not a.no_check:
        check_numbers(ctx)


if __name__ == "__main__":
    main()
