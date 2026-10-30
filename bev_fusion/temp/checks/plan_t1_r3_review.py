"""one-off: record the t1 stop / fix / restart decision and the memory-path review in plan.md 7.4"""
p = "plan.md"
s = open(p, encoding="utf-8").read()
old = "### 7.4 Temporal fusion: owner review and repairs (2026-10-26, Mirko: this session owns the temporal fusion code)\n"
assert s.count(old) == 1
new = old + (
    "- Mirko 2026-10-26 ~08:40: \"If we are training a buggy system, stop it, fix bugs, restart\". With the parent chat: t1_r0 (window-only, 2ddf511-era "
    "code, 92 % done) finishes as the labelled baseline; the queued round 1 (pinned to 2ddf511: no memory training, no write mask) was cancelled; "
    "instead ONE combined run t1_r3_combined (seq_len 3, batch 2 x accum 3, seq_bptt, object_direction, refine, radar cache) pinned to the HEAD "
    "at about 09:10.\n"
    "- Memory-path review for that run: (A) BUG fixed 5f401f8 - the confidence input conf / w_max stayed <= 0.25 in training with w_max 8 (at most 2 "
    "writes per 3-keyframe clip) but reaches 1.0 on a long drive: w_max 2 now (saturates inside a clip); (B) obs_mass reaches the cached training path; "
    "(C) the carried step and warm / cold validation are right; (D) seq_bptt recommended: with detach the write path (static weight of the unsupervised "
    "velocity head) gets no gradient and moving objects stay in the static memory as ghosts; (F) evaluation/eval_memory.py reports warm vs cold only in "
    "the current camera view, where the memory matters least - a meter on the cells outside the current view (map GT exists there) is needed, else the "
    "test can wrongly show no gain.\n")
s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8").write(s)
print("ok")
