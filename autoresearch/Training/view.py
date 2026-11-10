"""What one run looks like, for every reader of a run (the commands in cli.py, the class in api.py): the summary (state, step, ETA, loss, gradient, health findings, kill rules), the
detail text of one training and the speed row. They read the run directory through runs.Run and nothing else."""
import json
import re
import statistics
import time

from . import health, modules
from .health import Finding
from .kill import load_rules
from .runs import fmt_dur


def summary(run, ref=None, extra_rules=(), since_step=None):
    st = run.state()                                          # read first and once: the fields of one summary agree (a `.done` that appears meanwhile cannot make a running run 100 %)
    findings = health.check(run, ref, since_step)
    rules, problems = load_rules(run, extra_rules)
    findings += [Finding("alert", "kill_rule", p) for p in problems]
    findings += [Finding("warn", "bad_config", p) for p in run.config_problems]
    kills = [(r, *r.evaluate(run)) for r in rules]
    findings += [Finding("warn", "kill_key", f"kill rule {r.text!r}: the run passed step {r.step} and never logged {r.key!r}")
                 for r, st, _ in kills if st == "unknown"]
    loss = run.last(run.loss_key)
    ready = run.ready_pct(st)
    return dict(run=run.name, state=st, phase=run.phase(st), step=run.step, total=run.total, pct=ready, ready_pct=ready, ready_at=run.ready_at(st),
                s_per_step=run.s_per_step(), eta_s=run.eta_s(), loss=loss, grad=modules.grad_trend(run), skipped=run.skipped_steps()[0], data=modules.data_of(run),
                restarts=run.restarts, findings=[f._asdict() for f in findings],
                kill=[dict(rule=r.text, status=s, value=v) for r, s, v in kills]), findings, kills


def speed_row(run, skip=10):
    """{s_step, data_share, gpu_busy, gpu_gb, loss, step} of one run for the speed table: the median interval between the logged rows (the first `skip` rows are the warm-up), the
    probe's data wait over step time when the run logged it, the GPU busy share from profile.txt when there is one, the peak GPU allocation, the loss at the end"""
    rows = [r for r in run.train if "t" in r][skip:]
    gaps = [(b["t"] - a_["t"]) / (b["step"] - a_["step"]) for a_, b in zip(rows, rows[1:]) if b["step"] > a_["step"] and b["t"] > a_["t"]]
    pr = [r for r in rows if r.get("t_step") and "t_data" in r]
    busy = None
    try:
        with open(run.p("profile.txt"), encoding="utf-8") as f:
            m = re.search(r"GPU kernels busy: .*? = (\d+) %", f.read())
        busy = int(m.group(1)) if m else None
    except OSError:
        pass
    return dict(s_step=statistics.median(gaps) if gaps else None, step=run.step, loss=run.last(run.loss_key, 5),
                data_share=statistics.median(r["t_data"] / r["t_step"] for r in pr) if pr else None,
                gpu_busy=busy, gpu_gb=max([r["gpu_gb"] for r in run.train if "gpu_gb" in r], default=None))


def clock(t):
    """'HH:MM' of a unix time, with the day when it is not today; '-' for none"""
    if not t:
        return "-"
    try:
        lt, now = time.localtime(t), time.localtime()
        return time.strftime("%H:%M", lt) if lt[:3] == now[:3] else time.strftime("%a %H:%M", lt)
    except (OverflowError, OSError, ValueError):               # a step total of 10**30 gives a time no clock has
        return "-"


def events_of(run, n=5):
    """the last n events that `watch` wrote for this run (training_events.jsonl; a damaged line is skipped)"""
    out = []
    try:
        with open(run.p("training_events.jsonl"), encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if isinstance(e, dict):
                    out.append(e)
    except OSError:
        return []
    return out[-n:]


def _num(x, fmt="%.3g"):
    return "-" if x is None else fmt % x


def detail_text(run, row):
    """what a person reads about one training: phase and ETA, the findings and kill rules, the last evaluation, the data, the gradient norm, the skipped steps, the loss groups, the modules,
    the time split, the incident dumps and the last events of `watch` -> text (row = what `summary` returns)"""
    s, findings, kills = row
    out = []
    p = out.append
    if s["ready_pct"] is not None:
        p(f"  phase {s['phase']}, ready {s['ready_pct']:.0f} %" + (f", ETA {fmt_dur(s['eta_s'])} (last step about {clock(s['ready_at'])})" if s["ready_at"] else ""))
    else:
        p(f"  phase {s['phase']}, the number of steps is not known (no `steps` in config.json): no percentage")
    for f in findings:
        p(f"  [{f.level:<5}] {f.code}: {f.text}")
    for r, st, v in kills:
        p(f"  [kill ] {r.text}: {st}" + (f" (value {v:.4g})" if v is not None else ""))
    if run.evals:
        p("  last eval step %d: %s" % (run.evals[-1]["step"], ", ".join(
            f"{k[4:]} {v:.4g}" for k, v in run.evals[-1].items() if k.startswith("val.") and "last" not in k)))
    if run.restarts:
        p(f"  the stream was resumed {run.restarts} time(s); the later attempt of each step is used")
    if s.get("data"):
        p(f"  data: {s['data']}")
    g = s.get("grad")
    if g:
        p(f"  gradient norm: first 10 rows {_num(g['first'])}, now {_num(g['now'])} (median of the finite values in the last {modules.WINDOW} rows"
              + (f"; x{g['ratio']:.1f} against the rows before" if g["ratio"] else "") + f"), peak {_num(g['peak'])}" + ("" if g["peak_step"] is None else f" at step {g['peak_step']}")
              + (f"; {g['nonfinite_recent']} of those rows are NOT finite" if g["nonfinite_recent"] else ""))
    skipped = run.skipped_at()
    if skipped:
        n_skip, in_row = run.skipped_steps()
        shown = ", ".join(str(x) for x in skipped[:8]) + (f", ... (last {skipped[-1]})" if len(skipped) > 8 else "")
        p(f"  non-finite: {n_skip} step(s) skipped at {shown} (up to {in_row} in a row)")
    groups = [k for k in run.keys() if k.startswith("group.")]
    if groups:
        p("  loss groups (mean of the first 10 logged values -> the last 10):")
        for k in groups:
            v = [x for _, x in run.series(k)]
            p(f"    {k:<28}{_num(sum(v[:10]) / len(v[:10]))} -> {_num(sum(v[-10:]) / len(v[-10:]))}")
    mt = modules.module_table(run)
    if mt:
        p(f"  modules (median of the last {modules.WINDOW} rows; grad x = against the rows before; update = |dw| / |w| per step):")
        for m in mt:
            x = "-" if m["grad"] is None or not m["grad_base"] else "%.1f" % (m["grad"] / m["grad_base"])
            p(f"    {m['module']:<24}grad {_num(m['grad']):>9}  x{x:>5}  update {_num(m['upd']):>9}" + ("   (frozen on purpose)" if m["frozen"] else ""))
    tm = modules.timing(run)
    if tm:
        p(f"  time: {tm['t_step']:.2f} s per step, {tm['t_data']:.2f} s of it waiting for the data ({100 * tm['share']:.0f} %)")
    inc = modules.incidents(run)
    if inc:
        p(f"  incidents ({len(inc)}; the dump has the batch ids, replay it with the trainer's debug mode):")
        for i in inc[-3:]:
            p(f"    step {i['step']}: {i['kind']} ({i['file']})")
    ev = events_of(run, 5)
    if ev:
        p("  last events of `watch` (training_events.jsonl):")
        for e in ev:
            t_ = e.get("t") if isinstance(e.get("t"), (int, float)) else 0
            p(f"    [{time.strftime('%m-%d %H:%M:%S', time.localtime(t_))}] {e.get('kind', 'info')}: {e.get('text', '')}")
    if not run.timed() and run.train:
        p("  rows have no time `t` (trainer older than 2026-10-31): speed is the trainer's average, no slowdown check")
    return "\n".join(out)
