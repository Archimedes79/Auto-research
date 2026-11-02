"""stage A: move the three misplaced pieces of the slim change to where they belong, in the COMMITTED train.py (h_train.py);
stage B: the improvements the critic asked for, applied to the corrected committed text AND to the working-tree text (w_train_before.py).
Every replacement must match exactly once, else the script stops."""
import re
import sys

SP = sys.argv[1]


def once(t, old, new):
    assert t.count(old) == 1, (t.count(old), old[:90])
    return t.replace(old, new)


def grab(t, pattern, flags=re.M):
    m = re.search(pattern, t, flags)
    assert m, pattern
    return m.group(0)


GUARD = r'^        if ck\.get\("slim"\) and step < cfg\.steps:\n            raise SystemExit\(.*\n                             f"continue training; start a new run name"\)\n'
BLOCK_2B = r'^        # 2b slim_at_end \(the default\).*?the EMA weights survive the slimming"\n'

NEW_FUNC = '''def refuse_slim_resume(ck, step, cfg, name):
    """a slim checkpoint (a finished run: no optimiser state) evaluates and warm-starts but cannot continue training: a resume below the
    configured end is refused, at the end (a re-invocation of a finished run) it is accepted"""
    if ck.get("slim") and step < cfg.steps:
        raise SystemExit(f"{name} is a slim checkpoint (no optimiser state): it evaluates and warm-starts (init_from), it cannot "
                         f"continue training; start a new run name")


'''

NEW_2B = '''        # 2b slim_at_end (the default): a finished run keeps ONE checkpoint without optimiser state and RNG, and evaluation still loads it
        scfg = dataclasses.replace(cfg, name="slimrun", slim_at_end=True)
        _, _, sema, _ = train(scfg, datasets=(base, base))
        sdir = os.path.join(tmp, "slimrun")
        assert sorted(f for f in os.listdir(sdir) if f.startswith("ckpt_")) == ["ckpt_00000004.pt"], os.listdir(sdir)
        sck = torch.load(os.path.join(sdir, "ckpt_00000004.pt"), weights_only=False)
        assert sck["slim"] and "opt" not in sck and "rng" not in sck and sck["step"] == 4
        assert os.path.getsize(os.path.join(sdir, "ckpt_00000004.pt")) < 0.75 * os.path.getsize(os.path.join(tmp, "selftest", ckpts[-1]))
        m3, l3, _ = build(scfg)
        o3 = torch.optim.AdamW(param_groups(m3, l3, scfg.lr, scfg.weight_decay), lr=scfg.lr)
        e3 = EMA(m3, scfg.ema)
        assert load_checkpoint(os.path.join(sdir, "ckpt_00000004.pt"), m3, l3, o3, e3)[0] == 4
        assert all(torch.equal(e3.shadow[k].cpu(), sema.shadow[k].cpu()) for k in sema.shadow), "the EMA weights survive the slimming"
        # a slim checkpoint is refused as a resume below the configured end and accepted at the end
        for end, refused in ((6, True), (4, False)):
            try:
                refuse_slim_resume(sck, 4, dataclasses.replace(scfg, steps=end), "slimrun")
                got = False
            except SystemExit:
                got = True
            assert got == refused, (end, got)
        # re-invoking the finished slim run, even with slimming off, trains nothing and does not turn it into a "full" checkpoint with an
        # empty optimiser
        train(dataclasses.replace(scfg, slim_at_end=False), datasets=(base, base))
        sck2 = torch.load(os.path.join(sdir, "ckpt_00000004.pt"), weights_only=False)
        assert sck2["slim"] and "opt" not in sck2 and sorted(f for f in os.listdir(sdir) if f.startswith("ckpt_")) == ["ckpt_00000004.pt"]
'''


def stage_a(t):
    blk = grab(t, r'^    if slim: .*\n        del ck\["opt"\], ck\["rng"\]\n        ck\["slim"\] = True\n')
    t = once(t, blk, "")                                                        # out of load_checkpoint
    t = once(t, '    tmp = path + ".tmp"\n    torch.save(ck, tmp)\n', blk + '    tmp = path + ".tmp"\n    torch.save(ck, tmp)\n')   # into save_checkpoint
    guard = grab(t, GUARD)
    t = once(t, guard, "")                                                      # out of the training loop
    anchor = '        step, _ = load_checkpoint(os.path.join(out_dir, name), model, loss, opt, ema, ck=ck)\n'
    t = once(t, anchor, anchor + guard)                                         # after the load of the resume checkpoint
    b2 = grab(t, BLOCK_2B, re.M | re.S)
    t = once(t, b2, "")                                                         # out of the stop-run test
    a2 = '        assert changed, "parameters changed"\n'
    return once(t, a2, a2 + b2)


def stage_b(t):
    guard = grab(t, GUARD)
    t = once(t, guard, '        refuse_slim_resume(ck, step, cfg, name)\n        was_slim = bool(ck.get("slim"))\n')
    t = once(t, "def supersede_log(out_dir, step):", NEW_FUNC + "def supersede_log(out_dir, step):")
    t = once(t, "    step, damaged = 0, []\n", "    step, damaged, was_slim = 0, [], False\n")
    t = once(t, "cfg, slim=cfg.slim_at_end)", "cfg, slim=cfg.slim_at_end or was_slim)")
    t = once(t, "prune_checkpoints(out_dir, 1 if cfg.slim_at_end else cfg.keep)", "prune_checkpoints(out_dir, 1 if (cfg.slim_at_end or was_slim) else cfg.keep)")
    b2 = grab(t, BLOCK_2B, re.M | re.S)
    return once(t, b2, NEW_2B)


h = open(f"{SP}/h_train.py", newline="").read()
w = open(f"{SP}/w_train_before.py", newline="").read()
r = stage_b(stage_a(h))
open(f"{SP}/r_train.py", "w", newline="").write(r)
open(f"{SP}/w_train_new.py", "w", newline="").write(stage_b(w))
print("ok", len(h), len(r), len(w))
