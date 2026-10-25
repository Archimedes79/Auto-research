"""Step E: rewrite absolute project paths in code and wrapper files (byte level, so encodings stay untouched).
Rules map the old layout (/mnt/c/ML-Experiment/<old>) to the new one (/mnt/c/ML-Experiment/bev_fusion_lane_center/<new>) for the separators / , \ and \\ .
Usage: mig_rewrite.py [--go]   -> prints counts; with --go writes the files."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mig_common import *  # noqa: E402,F401,F403

GO = "--go" in sys.argv
B = "bev_fusion_lane_center"
# (old components, new components) most specific first
RULES = [
    (["out", "loop2"], [B, "runs"]),
    (["out", "crn"], [B, "runs", "_loop1_crn"]),
    (["report_final"], [B, "reports", "source"]),
    (["loop2", "splits.json"], [B, "data", "splits.json"]),
    (["loop2", "STATE.md"], [B, "experiment_execution", "STATE.md"]),
    (["loop2"], [B, "scripts", "_archive", "reports_build"]),
    (["reports"], [B, "reports"]),
    (["scripts"], [B, "scripts"]),
    (["literature"], [B, "literature"]),
    (["data"], [B, "data"]),
    (["downloads"], [B, "downloads"]),
    (["restructure"], [B, "tools", "restructure"]),
    (["bundles"], [B, "_to_delete", "bundles"]),
    (["recon"], [B, "_to_delete", "recon"]),
    (["mail13"], [B, "_to_delete", "mail13"]),
]
SEPS = ["/", "\\", "\\\\"]
EXT = (".py", ".sh", ".ps1", ".md", ".txt")
ROOTS = [f"{R}/scripts", f"{R}/evaluation", f"{R}/tools", f"{R}/experiments", f"{R}/experiment_execution", f"{R}/literature/README.md"]
SKIP_DIRS = {"__pycache__", "loop17_recovery"}  # recovery tree is rewritten when its files are copied into place


def rewrite(b):
    # protect already rewritten paths
    for sep in SEPS:
        b = b.replace(f"ML-Experiment{sep}{B}".encode(), f"ML-EXPERIMENT_DONE{sep}{B}".encode())
    for old, new in RULES:
        for sep in SEPS:
            o = ("ML-Experiment" + sep + sep.join(old)).encode()
            n = ("ML-EXPERIMENT_DONE" + sep + sep.join(new)).encode()
            # whole-component match: next byte must not continue the component name
            out, i = [], 0
            while True:
                j = b.find(o, i)
                if j < 0:
                    out.append(b[i:])
                    break
                nxt = b[j + len(o) : j + len(o) + 1]
                if nxt.isalnum() or nxt == b"_":
                    out.append(b[i : j + len(o)])
                else:
                    out.append(b[i:j] + n)
                i = j + len(o)
            b = b"".join(out)
    return b.replace(b"ML-EXPERIMENT_DONE", b"ML-Experiment")


changed, seen, hits = 0, 0, 0
files = []
for root in ROOTS:
    if os.path.isfile(root):
        files.append(root)
        continue
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in SKIP_DIRS]
        for f in fn:
            if f.endswith(EXT):
                files.append(os.path.join(dp, f))
for p in files:
    try:
        b = open(p, "rb").read()
    except OSError:
        continue
    seen += 1
    if b"ML-Experiment" not in b:
        continue
    nb = rewrite(b)
    if nb != b:
        changed += 1
        if GO:
            open(p, "wb").write(nb)
print(f"files scanned {seen}, {'rewritten' if GO else 'would be rewritten'}: {changed}")
