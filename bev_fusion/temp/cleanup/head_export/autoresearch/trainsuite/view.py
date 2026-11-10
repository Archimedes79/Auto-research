"""What one run looks like, for every reader of a run (the older commands in cli.py, the class in api.py): the summary (state, step, ETA, loss, gradient, health findings, kill rules) and the
speed row. They read the run directory through runs.Run and nothing else."""
import re
import statistics

from . import health, modules
from .health import Finding
from .kill import load_rules


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
