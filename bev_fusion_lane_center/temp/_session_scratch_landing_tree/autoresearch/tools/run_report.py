"""run_report.py - one page per training run that a person can read and judge without the code.

    python autoresearch/tools/run_report.py runs/t10_dual_long_s0 [--vs runs/t10_dual_s0 runs/t10_dual_s1] [--id t10.dual_long_s0]
           [--note "text for the reading guide"] [--max-pictures 24] [--out DIR]
    python autoresearch/tools/run_report.py runs/t10_dual_long_s0 --ingest ~/Downloads/review_t10_dual_long_s0.json

Writes <run>/report/index.html (opens from the file system, no server): what the run is (state, step, resumes, the card's prediction and rule
from the ledger row), loss curves, the headline numbers next to the reference runs (--vs; the first one is the control the delta column uses),
cost, every picture and video found under <run>/eval/suite, and a review panel: per picture ok / wrong / unsure + a comment, an overall remark,
saved in the browser and exported with one button as review_<run>.json. --ingest copies such a file into <run>/report/review.json and prints
the counts and the ledger note to write. Pictures and videos are not made here: `bev_suite.py RUN report` (BEV) makes them first, or any
adapter of the Evaluation (`qual random`, `qual videos`). Standard library; matplotlib only for the curves (skipped without it).
The problem comes from AR_PROJECT as for every tool (ledger, run records)."""
import argparse
import csv
import glob
import html
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
AR = os.path.dirname(HERE)
sys.path[:0] = [HERE, AR]
import project as P  # noqa: E402


def fmt(v):
    return "-" if v is None else (f"{v:.4g}" if isinstance(v, float) else html.escape(str(v)))


def read_table(path, limit=None):
    """eval/table.txt: 'key   value' per line -> [(key, float)] (all lines, or the first `limit`)"""
    out = []
    if os.path.exists(path):
        for line in open(path, encoding="utf-8", errors="replace"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    out.append((parts[0], float(parts[-1])))
                except ValueError:
                    pass
            if limit and len(out) >= limit:
                break
    return out


def ledger_row(run_id):
    path = P.path("ledger")
    if run_id and os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("id") == run_id:
                    return r
    return {}


def lower_is_better(key, card):
    b = (card.get("better") or {}).get(key)
    if b is not None:
        return b in (False, "lower", "<")
    last = key.replace(":", "/").split("/")[-1]
    return last in set(P.cfg().get("lower_is_better", []))


def attempts(run_id, rundir):
    """(number of launcher attempts, [steps a resume started from]) from the run record and train.log"""
    rec = os.path.join(P.path("run_records"), run_id) if run_id else ""
    n = None
    try:   # the record counts the earlier attempts itself (a restarted launcher overwrites run.log of the attempt before)
        n = int(json.load(open(os.path.join(rec, "run.json"), encoding="utf-8")).get("attempts_before", 0)) + 1
    except (OSError, ValueError):
        n = len(glob.glob(os.path.join(rec, "run.*.log"))) + 1 if rec and os.path.isdir(rec) else None
    resumes = []
    try:
        for line in open(os.path.join(rundir, "train.log"), encoding="utf-8", errors="replace"):
            if line.startswith("resumed from"):
                resumes.append(int(line.rsplit(" ", 1)[-1]))
    except (OSError, ValueError):
        pass
    return n, resumes


def curves(run, out_png):
    try:
        from Training import curves as C
        C.plot([run], C.default_keys(run), out_png)
        return True
    except Exception as e:  # matplotlib missing or a run without a log: the page says so
        print("curves skipped:", type(e).__name__, e)
        return False


def data_block(rundir):
    """Which data, GT, code and weights made the numbers (the card's provenance block), for each stored evaluation (val, and train when evaluated):
    enough to find the same frames again; a value the evaluation did not record says so instead of staying silent."""
    rows = []
    for label, name in (("val", "card.json"), ("train", "card_train.json")):
        try:
            c = json.load(open(os.path.join(rundir, "eval", name), encoding="utf-8"))
        except (OSError, ValueError):
            continue
        pv = c.get("provenance") or {}
        d, gt, sub, ev = pv.get("data") or {}, pv.get("gt") or {}, pv.get("subject") or {}, pv.get("evaluator") or {}
        use = d.get("use") or {}
        tg = sub.get("train_gt") or {}
        nr = "not recorded"
        rows.append((label, [("split rule", use.get("split_rule") or d.get("split") or nr),
                             ("scenes", f"{use.get('scenes_evaluated', nr)} evaluated of {use.get('scenes_available', nr)} available of {use.get('scenes_official', nr)} in the official split"),
                             ("frames", f"{use.get('frames_evaluated', c.get('frames', nr))}, list hash {d.get('frame_hash', nr)}, raw data {d.get('dataset', nr)} (manifest {d.get('manifest_hash', nr)})"),
                             ("GT scored against", f"version {gt.get('version', nr)}, signature {gt.get('signature', nr)}"),
                             ("GT the model was trained on", f"version {tg.get('version', nr)}, signature {tg.get('signature', nr)}"),
                             ("evaluator", f"code hash {ev.get('code_hash', nr)}, commit {ev.get('commit', nr)}, dirty: {ev.get('dirty', nr)}"),
                             ("weights", f"{sub.get('ckpt', nr)} (hash {sub.get('ckpt_hash', nr)}), step {sub.get('step', nr)}, seed {pv.get('seed', nr)}"),
                             ("scene names", ", ".join(use.get("scene_names", [])) or nr)]))
    if not rows:
        return ""
    out = ["<h2>Data used (traceable)</h2>"]
    for label, items in rows:
        out.append(f"<h3 style='font-size:14px'>{label}</h3><table>" + "".join(f"<tr><th class='l'>{html.escape(k)}</th><td class='l' style='max-width:1000px'>{html.escape(str(v))}</td></tr>" for k, v in items) + "</table>")
    return "".join(out)


def gap_block(rundir, card, val_path=None, train_path=None):
    """Training against validation: the same numbers on frames the network was trained on and on frames it never saw (eval/table_train.txt next to
    eval/table.txt). The training number is what this architecture, data and schedule can FIT (EMA weights, no augmentation, cold memory), the gap to
    the validation number is what is lost in generalisation. Only for runs evaluated on both splits (`bev_suite.py RUN generalisation`)."""
    tr = read_table(train_path or os.path.join(rundir, "eval", "table_train.txt"))
    va = dict(read_table(val_path or os.path.join(rundir, "eval", "table.txt")))
    if not tr or not va:
        return ""
    rows = []
    for key, t in tr:
        if key not in va:
            continue
        v = va[key]
        gap = t - v
        worse = (gap < 0) if lower_is_better(key, card) else (gap > 0)                  # training better than validation = the usual overfit direction
        rows.append(f"<tr><td class='l'>{html.escape(key)}</td><td>{fmt(t)}</td><td>{fmt(v)}</td><td>{gap:+.4f}</td>"
                    f"<td>{(100 * gap / t if t else float('nan')):+.1f} %</td><td>{'train better' if worse and abs(gap) > 1e-9 else ''}</td></tr>")
    return ("<h2>Training against validation (generalisation)</h2><table><tr><th class='l'>metric</th><th>train</th><th>val</th><th>train - val</th><th>relative</th><th></th></tr>"
            + "".join(rows) + "</table><p class='note'>train = frames the network was trained on (what it can fit: the practical ceiling of this architecture, data and schedule); "
            "val = scenes it never saw; the gap is the generalisation loss. Both with the EMA weights, no augmentation and an empty memory, same number of frames "
            "spread over the scenes of each split. The TEST split is not used here (it needs a logged row under a recorded decision). Caveat: the frozen image heads "
            "(pre-trained DualBEV / BEVFusion) saw the nuScenes train scenes too, so part of a train-val gap can come from them and not from the trainable network.</p>")


GAP = {}                                           # explicit table paths of the train-vs-val block (bev_suite generalisation), else eval/table.txt and eval/table_train.txt
OPS = {"fps": 10.0, "util": 0.3, "extra": []}      # set from the command line: frame rate, share of the peak a real accelerator reaches, fixed parts [(name, GOPs)]


def ops_block(rundir):
    """TOPS the network needs: operations per frame (2 x the MACs the suite counted with torch, + the fixed parts named by --extra-gops) x frame rate,
    and the peak an accelerator needs at the assumed utilisation. Counted: convolutions, matrix products, attention; NOT counted: element-wise ops,
    norms, interpolation, grid_sample, scatter, custom CUDA ops, numpy code - so it is a lower bound of the work, and memory traffic is not in it."""
    found = sorted(glob.glob(os.path.join(rundir, "eval", "suite", "quant", "*", "manifest.json")), key=os.path.getmtime)
    ops = None
    for f in reversed(found):
        try:
            ops = (json.load(open(f, encoding="utf-8")).get("cost") or {}).get("ops")
        except (OSError, ValueError):
            ops = None
        if ops:
            break
    if not ops or not ops.get("gmacs"):
        return "<h2>Compute (TOPS)</h2><p class='note'>no operation count stored (Subject.ops of the evaluation suite: `quant run` writes it into the manifest).</p>"
    net = 2.0 * float(ops["gmacs"])
    parts = [("network, counted by torch (" + html.escape(str(ops.get("method", "?"))) + ")", net)] + [(html.escape(n), g) for n, g in OPS["extra"]]
    tot = sum(g for _, g in parts)
    need = tot * OPS["fps"] / 1000.0
    rows = "".join(f"<tr><td class='l'>{n}</td><td>{g:.1f}</td></tr>" for n, g in parts)
    return (f"<h2>Compute (TOPS)</h2><table><tr><th class='l'>part</th><th>GOPs per frame</th></tr>{rows}<tr><th class='l'>total</th><th>{tot:.1f}</th></tr></table>"
            f"<p><b>{need:.2f} TOPS</b> sustained at {OPS['fps']:g} frames per second; at {100 * OPS['util']:.0f} % utilisation of the peak an accelerator needs about "
            f"<b>{need / OPS['util']:.1f} TOPS</b> (8-bit). 1 MAC = 2 operations.</p>"
            f"<p class='note'>{html.escape(str(ops.get('note', '')))}. Fixed parts named on the command line are figures from the code base or papers, not measured by this run.</p>")


def pictures(rundir, limit):
    root = os.path.join(rundir, "eval", "suite")
    found = [p for ext in ("png", "jpg", "mp4") for p in glob.glob(os.path.join(root, "**", "*." + ext), recursive=True)
             if os.sep + "quant" + os.sep not in p]
    return sorted(found)[:limit]


CSS = """body{font-family:Segoe UI,Arial,sans-serif;margin:18px auto;max-width:1400px;color:#1b1b1b}h1{font-size:22px}h2{font-size:17px;margin-top:28px;border-bottom:1px solid #ccc}
table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #bbb;padding:3px 8px;text-align:right}th{background:#eee}td.l,th.l{text-align:left}
.banner{padding:8px 12px;margin:10px 0;border-left:6px solid #c2410c;background:#fdebdd}.ok{color:#15803d}.bad{color:#b91c1c}.note{color:#555;font-size:12px}
.card{display:inline-block;vertical-align:top;margin:6px;padding:6px;border:1px solid #ccc;width:440px}.card img,.card video{width:100%}
textarea{width:100%;height:42px}select{margin:2px 0}code{background:#f3f3f3;padding:1px 3px}button{padding:6px 12px;margin:4px 0}"""

JS = """const KEY='review:'+location.pathname+':'+document.title;const items=[...document.querySelectorAll('.card')];
function load(){try{const d=JSON.parse(localStorage.getItem(KEY)||'{}');items.forEach(c=>{const x=d[c.dataset.f];if(x){c.querySelector('select').value=x.mark;c.querySelector('textarea').value=x.comment}});
document.getElementById('overall').value=d._overall||'';document.getElementById('who').value=d._who||''}catch(e){}}
function collect(){const d={};items.forEach(c=>{d[c.dataset.f]={mark:c.querySelector('select').value,comment:c.querySelector('textarea').value}});
d._overall=document.getElementById('overall').value;d._who=document.getElementById('who').value;return d}
function save(){try{localStorage.setItem(KEY,JSON.stringify(collect()))}catch(e){}}
document.addEventListener('input',save);load();
function exportReview(){const d=collect();const out={run:document.title,exported:new Date().toISOString(),reviewer:d._who,overall:d._overall,
items:items.map(c=>({file:c.dataset.f,mark:d[c.dataset.f].mark,comment:d[c.dataset.f].comment}))};
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,1)],{type:'application/json'}));a.download='review_'+document.title+'.json';a.click()}"""


def build(rundir, vs, run_id, note, max_pic, out_dir):
    from Training.runs import Run
    run = Run(rundir)
    os.makedirs(out_dir, exist_ok=True)
    rel = lambda p: os.path.relpath(p, out_dir).replace(os.sep, "/")  # noqa: E731
    state, step, total = run.state(), run.step, run.total
    card = {}
    try:
        card = json.load(open(os.path.join(rundir, "eval", "card.json"), encoding="utf-8"))
    except (OSError, ValueError):
        pass
    row = ledger_row(run_id)
    n_att, resumes = attempts(run_id, rundir)
    h = [f"<h1>{html.escape(run.name)}</h1>"]
    h.append(f"<p>state <b>{state}</b>, step {step} of {total} ({100 * step // max(total, 1)} %)"
             + (f", {n_att} launcher attempt(s)" if n_att else "") + (f", resumed at step {', '.join(map(str, resumes))}" if resumes else "")
             + f"; page made {time.strftime('%Y-%m-%d %H:%M')}</p>")
    if state != "done":
        h.append(f"<div class='banner'><b>Not finished</b> ({state} at step {step} of {total}). Numbers of an unfinished run carry no verdict "
                 "(the learning-rate schedule has not run out); read them as a progress report.</div>")
    if card:
        sig = (card.get("protocol") or {}).get("gt_signature", "?")
        h.append(f"<p class='note'>numbers: split <b>{html.escape(str(card.get('split')))}</b>, {card.get('frames')} frames, GT signature "
                 f"<code>{html.escape(str(sig))}</code>, evaluated {html.escape(str(card.get('date')))}, code commit "
                 f"<code>{html.escape(str(card.get('commit')))}</code></p>")
    elif os.path.isdir(os.path.join(rundir, "eval")):
        h.append("<div class='banner'>No metric card in eval/: the numbers below may be from another evaluation.</div>")
    if note:
        h.append(f"<p class='note'>{html.escape(note)}</p>")
    if row:
        h.append("<h2>What was expected, and what came out</h2><table>")
        for k in ("hypothesis", "prediction", "rule", "result", "decision", "status", "exit"):
            if row.get(k):
                h.append(f"<tr><th class='l'>{k}</th><td class='l'>{html.escape(row[k])}</td></tr>")
        h.append("</table>")
    png = os.path.join(out_dir, "curves.png")
    h.append("<h2>Training curves</h2>" + (f"<img src='{rel(png)}' style='max-width:100%'>" if curves(run, png) else "<p class='note'>no curves (no matplotlib or no log)</p>"))
    # numbers
    cols = [(run.name, read_table(os.path.join(rundir, "eval", "table.txt")))] + [(os.path.basename(v.rstrip("/\\")), read_table(os.path.join(v, "eval", "table.txt"))) for v in vs]
    h.append("<h2>Numbers</h2>")
    if cols[0][1]:
        ref = dict(cols[1][1]) if len(cols) > 1 else {}
        h.append("<div style='max-height:640px;overflow:auto;display:inline-block'><table><tr><th class='l'>metric</th>" + "".join(f"<th>{html.escape(n)}</th>" for n, _ in cols) + ("<th>delta vs " + html.escape(cols[1][0]) + "</th>" if ref else "") + "</tr>")
        maps = [dict(t) for _, t in cols]
        for key, v in cols[0][1]:
            cells = "".join(f"<td>{fmt(m.get(key))}</td>" for m in maps)
            d = ""
            if ref and key in ref and v == v and ref[key] == ref[key]:
                diff = v - ref[key]
                good = (diff < 0) == lower_is_better(key, card)
                d = f"<td class='{'ok' if good else 'bad'}'>{diff:+.4f}</td>" if diff else "<td>0</td>"
            elif ref:
                d = "<td>-</td>"
            h.append(f"<tr><td class='l'>{html.escape(key)}</td>{cells}{d}</tr>")
        h.append("</table></div><p class='note'>green = better than the first reference, red = worse; a difference inside the noise band of the card "
                 "is a tie (decide.py gives the verdict, not this page). Full list: eval/table.txt.</p>")
    else:
        h.append("<p class='note'>no eval/table.txt yet: run the evaluation first.</p>")
    h.append(gap_block(rundir, card, GAP.get('val'), GAP.get('train')))
    h.append(data_block(rundir))
    try:
        lat = json.load(open(os.path.join(rundir, "eval", "latency.json"), encoding="utf-8")).get("latency", {})
        h.append("<h2>Cost</h2><table>" + "".join(f"<tr><th class='l'>{k}</th><td>{fmt(v)}</td></tr>" for k, v in lat.items() if isinstance(v, (int, float)) and "median" in k or k == "p90_ms") + "</table>"
                 f"<p class='note'>ms per frame as measured by the evaluation; valid only if it ran on an idle GPU (recorded: idle = {lat.get('idle', 'not recorded')}).</p>")
    except (OSError, ValueError):
        pass
    h.append(ops_block(rundir))
    # pictures + review
    pics = pictures(rundir, max_pic)
    h.append("<h2>Pictures and videos, with a review field each</h2>")
    if not pics:
        h.append("<p class='note'>none stored under eval/suite. Make some: BEV <code>wsl python3 evaluation/bev_suite.py RUN report</code>, "
                 "any Evaluation adapter <code>qual random 6</code> and <code>qual videos 1</code>.</p>")
    for p in pics:
        r = rel(p)
        media = f"<video src='{r}' controls preload='none'></video>" if p.endswith(".mp4") else f"<a href='{r}'><img src='{r}'></a>"
        h.append(f"<div class='card' data-f='{html.escape(r)}'>{media}<div class='note'>{html.escape(os.path.basename(p))}</div>"
                 "<select><option value=''>- not looked at -</option><option>ok</option><option>wrong</option><option>unsure</option></select>"
                 "<textarea placeholder='what do you see?'></textarea></div>")
    h.append("<h2>Overall remark</h2><p>Reviewer <input id='who' size='20'> </p><textarea id='overall' placeholder='your judgement of the run'></textarea>"
             "<br><button onclick='exportReview()'>Export review.json</button> <span class='note'>saved in this browser while you type; the button "
             "writes the file you hand back (<code>run_report.py RUN --ingest FILE</code>).</span>")
    page = (f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(run.name)}</title><style>{CSS}</style></head><body>"
            + "\n".join(h) + f"<script>{JS}</script></body></html>")
    path = os.path.join(out_dir, "index.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    return path, len(pics)


def ingest(rundir, src):
    d = json.load(open(src, encoding="utf-8"))
    name = os.path.basename(os.path.normpath(rundir))
    if d.get("run") not in (None, name):
        raise SystemExit(f"this review is of run '{d.get('run')}', not of '{name}': not filed")
    out = os.path.join(rundir, "report", "review.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out):
        os.replace(out, out + time.strftime(".%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}.old")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1)
    marks = [i.get("mark") or "-" for i in d.get("items", [])]
    cnt = {m: marks.count(m) for m in sorted(set(marks))}
    print(f"review of {d.get('run')} by {d.get('reviewer') or '?'}: {cnt}; overall: {(d.get('overall') or '')[:300]}")
    print("ledger note to write: ledger.py note <id> --result '... review by", d.get("reviewer") or "?", cnt, "...'")
    return out


def selftest():
    """a synthetic finished run and an unfinished one with a stored table: the page holds the banner, the numbers with a signed delta and the review
    fields; --ingest keeps the exported review"""
    import tempfile
    tmp = tempfile.mkdtemp(prefix="run_report_")

    def make(name, steps, done, val):
        d = os.path.join(tmp, name)
        os.makedirs(os.path.join(d, "eval", "suite", "qualitative"))
        json.dump({"steps": steps}, open(os.path.join(d, "config.json"), "w"))
        with open(os.path.join(d, "log.jsonl"), "w") as f:
            for s in range(1, 21):
                f.write(json.dumps({"step": s * 10, "t": time.time() - 1000 + s, "total": 5.0 / s, "lr": 1e-4, "grad_norm": 1.0}) + "\n")
        open(os.path.join(d, "train.log"), "w").write("step 200: total 0.25\nresumed from ckpt_00000100.pt at step 100\n")
        if done:
            open(os.path.join(d, ".done"), "w").close()
        open(os.path.join(d, "eval", "table.txt"), "w").write(f"seg_fov/seg/dt+0.00/walkway/all/all/iou@0   {val}\nseg_fov/det/mATE   0.50\n"
                                                              + "".join(f"seg_fov/extra/k{i}   {1.0 - i / 100:.3f}\n" for i in range(60)))         # more than 40 rows: none may be cut
        open(os.path.join(d, "eval", "suite", "qualitative", "frame_a.png"), "wb").write(b"\x89PNG\r\n")
        return d

    ref, run = make("ref", 200, True, 0.30), make("run", 400, False, 0.34)
    open(os.path.join(run, "eval", "table_train.txt"), "w").write("seg_fov/seg/dt+0.00/walkway/all/all/iou@0   0.50\nseg_fov/det/mATE   0.40\n"
                                                                  + "".join(f"seg_fov/extra/k{i}   {1.0 - i / 100 + 0.05:.3f}\n" for i in range(60)))
    qd = os.path.join(run, "eval", "suite", "quant", "eval_n1")
    os.makedirs(qd)
    json.dump({"cost": {"ops": {"gmacs": 100.0, "method": "measured", "note": "test"}}}, open(os.path.join(qd, "manifest.json"), "w"))
    path, n = build(run, [ref], "selftest.run", "a note", 10, os.path.join(run, "report"))
    page = open(path, encoding="utf-8").read()
    assert n == 1 and "Not finished" in page and "+0.0400" in page and "class='ok'" in page and "resumed at step 100" in page, page[:600]
    assert page.count("class='card'") == 1 and "exportReview" in page and "<select>" in page
    assert page.count("seg_fov/extra/k") >= 2 * 60 and "k59" in page, "the numbers and the gap table show every row, not the first 40"
    assert "Training against validation" in page and "train better" in page and "Compute (TOPS)" in page and "<b>2.00 TOPS</b>" in page
    rev = os.path.join(tmp, "review_run.json")
    json.dump({"run": "run", "reviewer": "x", "overall": "fine", "items": [{"file": "a.png", "mark": "ok", "comment": ""}, {"file": "b.png", "mark": "wrong", "comment": "c"}]}, open(rev, "w"))
    out = ingest(run, rev)
    assert json.load(open(out))["reviewer"] == "x"
    print("run_report selftest ok:", path)


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--vs", nargs="*", default=[], help="reference run directories (the first is the control of the delta column)")
    ap.add_argument("--id", default="", help="ledger / run record id (default: the run folder name with the first _ after the experiment prefix as a dot)")
    ap.add_argument("--note", default="", help="reading guide: what the numbers are scored on, caveats")
    ap.add_argument("--max-pictures", type=int, default=24)
    ap.add_argument("--fps", type=float, default=10.0, help="frame rate the TOPS estimate is for")
    ap.add_argument("--util", type=float, default=0.3, help="share of the peak TOPS a real accelerator reaches on this network")
    ap.add_argument("--extra-gops", nargs="*", default=[], metavar="NAME=GOPS", help="fixed parts the suite does not count, e.g. 'frozen image heads=71'")
    ap.add_argument("--out", default="")
    ap.add_argument("--ingest", default="", help="a review.json exported from the page")
    ap.add_argument("--val-table", default="", help="table of the validation numbers for the train-vs-val block (default eval/table.txt)")
    ap.add_argument("--train-table", default="", help="table of the train numbers (default eval/table_train.txt)")
    a = ap.parse_args()
    rundir = os.path.abspath(a.run)
    if a.ingest:
        print(ingest(rundir, a.ingest))
        return
    GAP.update(val=a.val_table or None, train=a.train_table or None)
    OPS.update(fps=a.fps, util=a.util, extra=[(x.rsplit("=", 1)[0], float(x.rsplit("=", 1)[1])) for x in a.extra_gops])
    name = os.path.basename(rundir.rstrip("/\\"))
    rid = a.id or name.replace("_", ".", 1)
    path, n = build(rundir, [os.path.abspath(v) for v in a.vs], rid, a.note, a.max_pictures, a.out or os.path.join(rundir, "report"))
    print(f"report: {path} ({n} pictures/videos)")


if __name__ == "__main__":
    main()
