p = "autogt/checks.py"
s = open(p, encoding="utf-8").read()
s = s.replace('''import collections
import math
import os
''', '''import collections
import hashlib
import math
import os
''', 1)
s = s.replace('''def _cell(grid, x, y):''', '''SRC_HASH = hashlib.sha1(open(__file__, "rb").read()).hexdigest()[:10]       # version of the checks: stored in every accumulator


def _cell(grid, x, y):''', 1)
s = s.replace('''        self.gen, self.st''', '''        self.gen, self.st''')
s = s.replace('''        self.grid, self.st = grid, collections.defaultdict(list)''', '''        self.grid, self.st = grid, collections.defaultdict(list)
        self.version = SRC_HASH''', 1)
open(p, "w", encoding="utf-8").write(s)

p = "autogt/report.py"
s = open(p, encoding="utf-8").read()
s = s.replace("from .checks import Checks, leave_one_radar_out, radar_fov  # noqa: E402", "from .checks import SRC_HASH, Checks, leave_one_radar_out, radar_fov  # noqa: E402")
s = s.replace('''        try:
            out = pickle.load(open(pk, "rb"))
            if out[1] is not None:
                out[1].written = 0                                  # nothing was written in THIS run
            return out
        except Exception:                                           # noqa: BLE001  (a half-written file: recompute)
            pass''', '''        try:
            out = pickle.load(open(pk, "rb"))
            ck_old = out[1]
            # reuse only a pickle that fits THIS run: same stride, a sample exactly where one is wanted, same version of the checks
            # (pickles of before 2026-10-30 carry no version: accepted, their checks.py is the one that has not changed since)
            fits = (out[3] is None and ck_old is not None and ck_old.per_scene[name]["keyframes"] == len(range(0, len(gen.keyframes(name)), stride))
                    and (out[2] is not None) == want_sample and getattr(ck_old, "version", SRC_HASH) == SRC_HASH)
            if fits:
                ck_old.written = 0                                  # nothing was written in THIS run
                return out
        except Exception:                                           # noqa: BLE001  (a half-written file: recompute)
            pass''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
