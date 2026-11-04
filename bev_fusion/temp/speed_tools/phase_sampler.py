"""throw-away (2026-11-03, Mirko: 'Yes, time it'): where does the wall time of a REAL t20 training step go? No code of the trainer is changed: this driver imports the code
snapshot of t20.base_s0 (runs/t20_base_s0/code, commit bd7eb0181b), runs `train()` of the t20 config for N optimiser steps and samples the main thread's Python stack every
~3 ms (a wall-clock sampler, like py-spy). Every sample is put into a section by the frames of the trainer (data wait, model forward, loss, backward, optimiser, EMA, copies to the GPU,
read-backs ...) and, inside it, by the innermost repo frame. Time inside a blocking call (`.item()`, a copy, a mask index, the backward) is counted where the main thread waits.
No checkpoint is written. Usage: python phase_sampler.py <code dir> <config> <tag> <steps> <warm-up steps> [<init checkpoint>]
  fresh init (no checkpoint)        = the state of the first 1000 steps of a training
  <init checkpoint> (a trained one) = the state of the last 1000 steps (lr 1e-6, so the weights hardly move): tests whether a trained model makes the step slower"""
import collections
import dataclasses
import json
import linecache
import os
import sys
import threading
import time

CODE, CFG, TAG, N, WARM = os.path.abspath(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
INIT = sys.argv[6] if len(sys.argv) > 6 else ""
sys.path.insert(0, CODE)
import train as T                                   # noqa: E402  (the pinned snapshot, with its trainkit next to it)
import trainkit.checkpoint as CK                    # noqa: E402

CK.save_checkpoint = lambda *a, **k: None           # nothing is written at the end of this diagnostic
MAIN = threading.main_thread().ident
RULES = {                                           # (file, function) -> [(substring of the source line, section)]; a deeper frame that matches overrides
    ("loop.py", "fit"): [("next_batch()", "data wait"), ("optimizer_step", "optimiser step (unscale, clip, AdamW)"), ("ema.update", "EMA update"),
                         ("step_fn(", "step glue"), ("hooks.", "hooks / log"), ("zero_grad", "zero_grad"), ("stop_check", "stop check")],
    ("train.py", "run_clip"): [(".backward(", "backward"), ("forward_loss(", "forward+loss glue"), ("_pack(", "read-back (pack)"), ("_unpack(", "read-back (unpack)"),
                               ("isfinite", "read-back (finite check)"), ("clip_step(", "clip_step (ego tensors)"), ("flip_batch", "augment flip"), ("detach_tree", "detach state")],
    ("train.py", "forward_loss"): [("sample_targets(", "targets: sample_targets"), ("batch_ego(", "ego prep"), ("to_device(", "batch to GPU"), ("res = model(", "model forward"),
                                   ("out = loss(", "loss"), ("aux_loss", "radar aux loss"), ("consistency", "consistency"), ("check_inputs", "contract check"),
                                   ("check_outputs", "contract check"), ("group_gradient_cosines", "diag")],
}
state = {"step": 0, "ends": [], "t0": 0.0, "t1": 0.0}
sec_t, sub_t, line_t = collections.defaultdict(float), collections.defaultdict(float), collections.defaultdict(float)
stop = threading.Event()


def classify(fr):
    st = []
    while fr is not None:
        st.append((fr.f_code.co_filename, fr.f_code.co_name, fr.f_lineno))
        fr = fr.f_back
    st.reverse()                                    # outermost first
    sec, inner = "other (not in the trainer's frames)", None
    for path, name, ln in st:
        if not path.startswith(CODE):
            continue
        base = os.path.basename(path)
        txt = linecache.getline(path, ln).strip()
        for key, label in RULES.get((base, name), ()):
            if key in txt:
                sec = label
                break
        inner = (base, name, ln, txt)
    return sec, inner


def sampler():
    last = time.perf_counter()
    while not stop.is_set():
        time.sleep(0.003)
        fr = sys._current_frames().get(MAIN)
        now = time.perf_counter()
        dt, last = now - last, now
        if fr is None or state["step"] < WARM:
            continue
        sec, inner = classify(fr)
        sub = f"{inner[0]}:{inner[1]}" if inner else "-"
        sec_t[sec] += dt
        sub_t[(sec, sub)] += dt
        if inner:
            line_t[f"{inner[0]}:{inner[1]}:{inner[2]}  {inner[3][:78]}"] += dt


orig_begin, orig_end = T.ProbeHooks.begin_step, T.ProbeHooks.end_step
thread = threading.Thread(target=sampler, daemon=True)


def begin_step(self, n):
    if not thread.is_alive() and not stop.is_set():
        thread.start()                              # after the data workers were forked (fit calls iter(batches) before the first step)
    if state["step"] == WARM:
        state["t0"] = time.perf_counter()
    return orig_begin(self, n)


def end_step(self):
    state["step"] += 1
    state["ends"].append(time.perf_counter())
    return orig_end(self)


T.ProbeHooks.begin_step, T.ProbeHooks.end_step = begin_step, end_step
cfg = T.read_config(CFG)
changes = dict(name=f"timing_phase_{TAG}", steps=N, eval_every=0, save_every=0, init_from=INIT, slim_at_end=True)
if INIT:
    changes.update(lr=1e-6, warmup=0)
cfg = dataclasses.replace(cfg, **changes)
print(f"phase sampler: tag {TAG}, {N} steps, the first {WARM} not counted, code {CODE}, init {INIT or 'fresh'}, wall clock start {time.strftime('%H:%M:%S')}", flush=True)
T.train(cfg)
stop.set()
t_end = time.perf_counter()
ends = state["ends"]
win = ends[WARM - 1:]                               # end of step WARM ... end of the last step
per = [b - a for a, b in zip(win, win[1:])]
n_win = len(per)
total = sum(sec_t.values())
print(f"\nWINDOW {time.strftime('%H:%M:%S', time.localtime(time.time() - (t_end - ends[WARM - 1])))} {time.strftime('%H:%M:%S')}")
print(f"steps in the window: {n_win}   s/step median {sorted(per)[n_win // 2]:.2f}  mean {sum(per) / n_win:.2f}  min {min(per):.2f}  max {max(per):.2f}   (sampled time {total / n_win:.2f} s per step)")
print("\nsection                                          s/step   share")
for k, v in sorted(sec_t.items(), key=lambda kv: -kv[1]):
    print(f"  {k:48s} {v / n_win:6.3f}  {100 * v / total:5.1f} %")
for sec in [k for k, v in sorted(sec_t.items(), key=lambda kv: -kv[1])][:6]:
    print(f"\n  inside '{sec}': innermost trainer frame")
    for (s, sub), v in sorted(sub_t.items(), key=lambda kv: -kv[1]):
        if s == sec and v / n_win > 0.01:
            print(f"      {sub:46s} {v / n_win:6.3f}")
print("\ntop source lines by wall time (s per step; the line where the main thread was when sampled)")
for k, v in sorted(line_t.items(), key=lambda kv: -kv[1])[:30]:
    print(f"  {v / n_win:6.3f}  {k}")
json.dump({"tag": TAG, "steps": n_win, "per_step": per, "sections": {k: v / n_win for k, v in sec_t.items()},
           "subs": {f"{s} | {sub}": v / n_win for (s, sub), v in sub_t.items()}, "lines": {k: v / n_win for k, v in line_t.items()}},
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"phase_{TAG}.json"), "w"), indent=1)
