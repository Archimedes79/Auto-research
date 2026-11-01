import sys
p, t = sys.argv[1], sys.argv[2]
s = open(p, encoding="utf-8").read()
def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:70], s.count(old))
    s = s.replace(old, new)
rep("""  * GPU         --gpu: waits for a free GPU slot (project.json gpu_slots, default 2; slots are directories under <run_records>/.gpu_slots taken""",
    """  * GPU         --gpu: waits for a free GPU slot (project.json gpu_slots, default 2; slots are directories under project.json gpu_slots_dir
                (relative to the problem folder; problems that share one GPU name the same folder), default <run_records>/.gpu_slots, taken""")
rep("""            if st == "done":
                continue
            if st in ("failed", "gpu_busy", "blocked", "timeout", "stalled"):""",
    """            if st in ("done", "ok"):  # ok = exit 0 whose done flag could not be written
                continue
            if st in ("failed", "gpu_busy", "blocked", "timeout", "stalled"):""")
rep('''    def __init__(self, records, n, run_id, run_dir):
        self.base, self.n, self.id, self.run_dir, self.held = os.path.join(records, ".gpu_slots"), n, run_id, run_dir, None''',
'''    def __init__(self, records, n, run_id, run_dir, base=None):
        self.base, self.n, self.id, self.run_dir, self.held = base or os.path.join(records, ".gpu_slots"), n, run_id, run_dir, None''')
rep('''def kill_tree(p):''', '''def slot_dir(root, cfg, records):
    """the folder of the GPU slots: project.json gpu_slots_dir (relative to the problem folder; shared by the problems of one machine), else per problem"""
    d = cfg.get("gpu_slots_dir")
    return os.path.normpath(os.path.join(root, d)) if d else os.path.join(records, ".gpu_slots")


def kill_tree(p):''')
rep('''    """one attempt: returns (exit code, text the attempt wrote, pid); the heartbeat is touched while the command runs"""''',
    '''    """one attempt: returns (exit code, text the attempt wrote, pid, killed by the timeout); the heartbeat is touched while the command runs"""''')
rep('''        t0, last = time.time(), 0
        rc = None
        while rc is None:''', '''        t0, last = time.time(), 0
        rc, timed_out = None, False
        while rc is None:''')
rep('''                kill_tree(p)
                rc = 124
                lg.write''', '''                kill_tree(p)
                rc, timed_out = 124, True
                lg.write''')
rep('''    return rc, out, p.pid


def main():''', '''    return rc, out, p.pid, timed_out


def main():''')
rep('''    slot = GpuSlot(records, a.gpu_slots, a.id, d) if (a.gpu or a.gpu_max_mb) else None''',
    '''    slot = GpuSlot(records, a.gpu_slots, a.id, d, base=slot_dir(root, cfg, records)) if (a.gpu or a.gpu_max_mb) else None''')
rep('''        rc, attempt, tail = 1, 0, ""''', '''        rc, attempt, tail, timed_out = 1, 0, "", False''')
rep('''                rc, out, pid = run_once(cmd, a.cwd, os.path.join(d, "run.log"), hb, a.timeout * 60)''',
    '''                rc, out, pid, timed_out = run_once(cmd, a.cwd, os.path.join(d, "run.log"), hb, a.timeout * 60)''')
rep('''            if rc == 0 or rc == 124 or not CUDA_RETRY.search(out):''', '''            if rc == 0 or timed_out or not CUDA_RETRY.search(out):''')
rep('''    status = "ok" if rc == 0 else "timeout" if rc == 124 else "failed"''', '''    status = "ok" if rc == 0 else "timeout" if timed_out else "failed"  # a command's own exit 124 is a failure, not the launcher's timeout''')
open(p, "w", encoding="utf-8", newline="").write(s)
s = open(t, encoding="utf-8").read()
rep('''def free_gpu_slots(records, n):
    """n minus the slot directories that are held by a live launcher (run.py GpuSlot is the semaphore; a stale directory does not count)"""
    base = os.path.join(records, ".gpu_slots")
    probe = RUN.GpuSlot(records, n, "tick", "")''', '''def free_gpu_slots(records, n, base=None):
    """n minus the slot directories that are held by a live launcher (run.py GpuSlot is the semaphore; a stale directory does not count)"""
    base = base or os.path.join(records, ".gpu_slots")
    probe = RUN.GpuSlot(records, n, "tick", "", base=base)''')
rep('''    free = free_gpu_slots(records, int(cfg.get("gpu_slots", 2)))''', '''    free = free_gpu_slots(records, int(cfg.get("gpu_slots", 2)), RUN.slot_dir(root, cfg, records))''')
open(t, "w", encoding="utf-8", newline="").write(s)
print("ok")
