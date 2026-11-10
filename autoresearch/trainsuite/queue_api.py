"""The queue of the training system, NOT part of the class `Training` (Mirko 2026-11-10: "Nimm die Queue hier erstmal raus"). `QueueMixin` adds the queue methods to a class that has the
methods of Training: `class Q(QueueMixin, Training)` with `METHODS = Training.METHODS + QueueMixin.QUEUE_METHODS`. Nothing in trainsuite uses it; the engine is trainqueue.py (the line, the runner,
holds, restarts). A pause and a held item forbid a direct `run` too, with or without this mixin (trainqueue.start_blocks, asked by Training.check / run); with the mixin `i` can be the number `queue_list` shows."""
import os

from . import health, modules
from . import trainqueue as TQ
from .api import TrainingError, _checked_plan, _machine, _map, _run_dir, _slot_ids, errors, shell
from .runs import Run


def _ordered(q):
    return sorted(q["items"], key=TQ.order)


def _states(c, items):
    """{id: state} of the items: state() reads files, so a call looks once"""
    return {it["id"]: TQ.state(c, it) for it in items}


def _running(c, items, states, only_trainings=False):
    """[(id, queue item or None, number or None)] of what runs now: the items of the queue first, then what holds a GPU slot without being in the queue (a run started by `run`,
    `run` or by hand). only_trainings leaves out the GPU jobs that have no run folder of their own (an evaluation, a prediction)"""
    out = [(it["id"], it, n) for n, it in enumerate(items, 1) if states[it["id"]] in ("running", "orphan")]
    queued = {it["id"] for it in items}
    for rid in _slot_ids(c):
        if rid not in queued and (not only_trainings or os.path.isdir(_run_dir(c, rid))):
            out.append((rid, None, None))
    return out


def resolve_item(c, i):
    """-> (id, queue item or None, number or None). i=None is the running training. A text is an id or a run folder name first; a number (an int, or a text of digits that is no
    id) is the place in `queue_list`; what is neither in the queue is a run of the records (no item, no number)"""
    items = _ordered(TQ.load(c))
    if i is None:
        running = _running(c, items, _states(c, items), only_trainings=True)
        return running[0] if running else (None, None, None)
    if isinstance(i, bool):
        raise TrainingError("i is a queue number, an id or a run folder name")
    if isinstance(i, str):
        for n, it in enumerate(items, 1):
            if it["id"] == i or TQ.folder(it["id"]) == TQ.folder(i):
                return it["id"], it, n
        known = next((d for d in sorted(os.listdir(c.records)) if TQ.folder(d) == TQ.folder(i)), None) if os.path.isdir(c.records) else None
        if known:
            return known, None, None
        if not i.isdigit():
            return i, None, None
        i = int(i)
    if isinstance(i, int):
        if not 1 <= i <= len(items):
            raise TrainingError(f"no queue item number {i}: the queue has {len(items)} item(s)")
        return items[i - 1]["id"], items[i - 1], i
    raise TrainingError("i is a queue number, an id or a run folder name")


def get_item(c, i):
    """one item of the queue of the project c, in full (queue_get)"""
    rid, it, n = resolve_item(c, i)
    if it is None:
        raise TrainingError(f"{i!r} is not in the queue")
    info = _item_info(c, it, n, _states(c, [it]))
    return dict(info, job=it.get("job"), kill=it.get("kill") or [], config=it.get("config"), composed=it.get("composed", ""), added=it.get("added"), by=it.get("by", ""),
                command=shell(TQ.with_gate(c, TQ.pinned_argv(it))))


def _item_info(c, it, n, states):
    rid = it["id"]
    rd = _run_dir(c, rid)
    step, total = TQ.last_step(rd)
    info = dict(number=n, id=rid, state=states[rid], step=step, total=total, after=list(it.get("after") or []), priority=it.get("priority", 5),
                note=it.get("note", ""), held=bool(it.get("held")), pin=(it.get("pin") or "")[:10], refused=it.get("refused", ""), gave_up=bool(it.get("gave_up")),
                grad=None, skipped=0, flags=[])
    if os.path.isdir(rd):
        try:
            run_ = Run(rd, "total")
            if run_.train:
                info["grad"], info["skipped"] = modules.grad_trend(run_), run_.skipped_steps()[0]
                info["flags"] = sorted({f.code for f in health.check(run_) if f.level != "info"})
        except Exception as e:  # noqa: BLE001  a damaged run folder must not break the view of the line
            info["flags"] = [f"unreadable:{type(e).__name__}"]
    return info


def _list_text(r):
    print(f"{'#':>2}  {'id':<24}{'state':<12}{'step/total':<13}{'grad':>9}  {'skip':>4}  flags / notes")
    for x in r["items"]:
        g = x["grad"]
        grad = "-" if not g else "non-finite" if g["now"] is None else "%.3g" % g["now"]
        more = ", ".join(filter(None, [",".join(x["flags"]), ("after " + ",".join(x["after"])) if x["after"] else "", "held" if x["held"] else "", x["refused"], x["note"]]))
        print(f"{x['number']:>2}  {x['id']:<24}{x['state']:<12}{'%d/%s' % (x['step'], x['total'] or '?'):<13}{grad:>9}  {x['skipped'] or '-':>4}  {more}")
    if not r["items"]:
        print("  the queue is empty")
    print(f"running: {r['running'] or '-'}   runner: {('pid %s' % r['runner']) if r['runner'] else 'not running'}   {'PAUSED' if r['paused'] else ''}")
    print("slots: " + "; ".join(f"{s['slot']}: {s['id'] or 'free'}" + (f" (loading {s['loading_s']} s)" if s.get("loading_s") is not None else "") for s in r["slots"]))
    lk = r["lock"]
    print("machine lock: " + (f"{lk['owner']} until {lk['end']} ({'exclusive' if lk.get('exclusive') is True else 'reservation'})" if lk else "free"))
    ram, gpu = r["ram"], r["gpu"]
    print(f"RAM: {ram['available_mb']} MB available (gate {ram['need_mb']} MB)   GPU memory: {gpu['used_mb']} MB in use (gate {gpu['gate_mb']})")
    if r["holds"]:
        print("on hold (the person's words clear a hold): " + ", ".join(h["id"] for h in r["holds"]))


QUEUE_TEXT = {"queue_list": _list_text, "queue_remove": lambda r: print(f"removed {r['removed']} from the queue"), "queue_start": lambda r: print(f"the runner runs (pid {r['runner']})")}


def _one_or_all(i, all_):
    if all_ and i not in (None, ""):
        raise TrainingError("an item or all=True, not both")
    if not all_ and i in (None, ""):
        raise TrainingError("name the item (i) or give all=True: without either nothing is held or resumed")


class QueueMixin:
    QUEUE_METHODS = ("queue_add", "queue_list", "queue_get", "queue_remove", "queue_start", "queue_hold", "queue_resume")

    @errors
    def queue_list(self):
        """the queue in the order it runs, with what the box looks like. -> {items: [{number, id, state, step, total, after, priority, held, grad, skipped, flags, ...}], running,
        runner (pid or 0), paused, slots, lock, ram, gpu, holds}"""
        c = self._c()
        q = TQ.load(c)
        items = _ordered(q)
        states = _states(c, items)
        infos = _map(lambda nx: _item_info(c, nx[1], nx[0], states), enumerate(items, 1))
        running = next(iter(_running(c, items, states)), (None,))[0]
        return dict(items=infos, running=running, runner=TQ.runner_alive(c), paused=bool(q.get("paused")), **_machine(c))

    @errors
    def queue_get(self, i):
        """one item: its place and state, the job it was made from, the composed config, and the exact command (the memory gate and the code pin included). -> dict"""
        return get_item(self._c(), i)

    @errors
    def queue_add(self, job, start=True):
        """check the job like `check`, put it in the queue and make sure the runner runs (start=False: not). A missing data build is refused here (a queued run would only fail when its
        turn comes). -> the item as queue_get shows it"""
        j = self._job(job)
        c = self._c(j)
        plan, problems, missing = _checked_plan(c, j, final=False)
        if missing:
            problems.append("missing data: " + ", ".join(m["name"] for m in missing) + " (build it first)")
        if problems:
            raise TrainingError("nothing queued: " + "; ".join(problems))
        TQ.add(c, plan, j["id"], j["after"], j["priority"], j["note"], extra=dict(job={k: v for k, v in j.items() if k != "project"}, kill=j["kill"]))
        if start:
            TQ.ensure_runner(c)
        return get_item(c, j["id"])                                        # the context of the job (its "project" may differ from this object's)

    @errors
    def queue_remove(self, i):
        """take an item out of the queue that has not started (or has ended); a running one is refused (it has to be stopped first). -> {removed}"""
        c = self._c()
        rid, it, _ = resolve_item(c, i)
        if it is None:
            raise TrainingError(f"{i!r} is not in the queue")
        TQ.drop(c, rid)
        return dict(removed=rid)

    @errors
    def queue_start(self):
        """make sure the runner runs: it starts the first startable item every 30 s, restarts a run whose launcher died with the machine, and ends after 10 idle minutes. -> {runner: pid}"""
        return dict(runner=TQ.ensure_runner(self._c()))

    @errors
    def queue_hold(self, words, i=None, all=False, reason="", wait=180.0):
        """stop the trainings (a checkpoint each, exit 75) and keep what waits from starting: `i` holds one, `all=True` holds ALL and pauses the runner (neither is an error, so a missing id
        never holds everything). `words` are the person's own words that ordered it (required). -> {id: outcome}"""
        c = self._c()
        _one_or_all(i, all)
        ids = () if all else (resolve_item(c, i)[0],)
        return TQ.hold(c, ids, all, words, reason, wait, out=lambda *a, **k: None)

    @errors
    def queue_resume(self, words, i=None, all=False, fresh_code=False):
        """the person's words clear the holds: `i` or `all=True` (every held training); each continues from its last checkpoint on the code of its first attempt (fresh_code: on the code of
        now, the hot-fix loop). -> {id: what happens}"""
        c = self._c()
        _one_or_all(i, all)
        ids = () if all else (resolve_item(c, i)[0],)
        return TQ.resume(c, ids, all, words, fresh_code, True, out=lambda *a, **k: None)

    # ---- a training has a place in the line (a pause or a held item binds a direct start too: Training.run / check ask the queue file, trainqueue.start_blocks)
    @errors
    def status(self, i=None, kill=(), ref=None, skip=10):
        """Training.status with the place in the queue: `number` and `queue` (the state of the item). -> as Training.status"""
        c = self._c()
        rid, it, n = resolve_item(c, i)
        out = super().status(rid, kill, ref, skip)
        out["number"], out["queue"] = n, (TQ.state(c, it) if it else None)
        if it and out.get("state") == "unknown":
            out["state"] = out["queue"]
        return out
