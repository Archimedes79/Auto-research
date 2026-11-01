"""repeat the GPU-slot race test of autoresearch/tools/selftest.py (defect 4) and print the state when it fails"""
import json
import os
import shutil
import sys
import tempfile
import threading
import time

sys.path.insert(0, "/mnt/c/Auto-research/autoresearch/tools")
import run as RUN  # noqa: E402

fails = 0
for rep in range(int(sys.argv[1]) if len(sys.argv) > 1 else 40):
    T = tempfile.mkdtemp()
    slots = f"{T}/runs/.race_slots"
    late = RUN.GpuSlot(f"{T}/runs", 1, "S.late", f"{T}/runs/S.late", base=slots)
    assert late.try_take()
    json.dump(dict(id="S.late", pid=999999, run_dir=f"{T}/runs/S.late", token=late.token), open(f"{slots}/0/run.json", "w"))
    sync = threading.Barrier(4)
    log = []

    class Racy(RUN.GpuSlot):
        judged = False

        def stale(self, s):
            r = super().stale(s)
            log.append((round(time.time() % 100, 3), self.id, "stale", r, RUN.record(s).get("id")))
            if not self.judged:
                self.judged = True
                sync.wait(10)
            return r

        def own(self, s):
            log.append((round(time.time() % 100, 3), self.id, "own"))
            super().own(s)
    racers = [Racy(f"{T}/runs", 1, f"S.race{i}", f"{T}/runs/S.race{i}", base=slots) for i in range(4)]
    real_rmtree = shutil.rmtree

    def slow_rmtree(p, *a, **k):
        name = threading.current_thread().name
        time.sleep(0.15 * int(name[4:]) if name.startswith("race") else 0)
        log.append((round(time.time() % 100, 3), name, "rmtree", p[-8:]))
        real_rmtree(p, *a, **k)
    took = {}
    ths = [threading.Thread(target=lambda g=g: took.update({g.id: g.try_take()}), name=f"race{i}") for i, g in enumerate(racers)]
    shutil.rmtree = slow_rmtree
    try:
        for th in ths:
            th.start()
        for th in ths:
            th.join()
    finally:
        shutil.rmtree = real_rmtree
    winner = [g for g in racers if took.get(g.id)]
    ok = len(winner) == 1
    if ok:
        for g in racers:
            if g is not winner[0]:
                g.release()
        late.release()
        ok = RUN.record(f"{slots}/0").get("id") == winner[0].id
    if not ok:
        fails += 1
        print(f"rep {rep}: FAIL took={took} record={RUN.record(f'{slots}/0')}")
        for e in log:
            print("   ", e)
    shutil.rmtree(T, ignore_errors=True)
print(f"{fails} failures")
