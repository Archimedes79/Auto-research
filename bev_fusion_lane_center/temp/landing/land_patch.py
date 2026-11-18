"""The text edits of the 07:00 - 08:30 landing (D117 / D119: archive tags, the S2 / S3 / jtok2 / jrecur removal; the lead's approval 2026-11-18 01:00), written as exact, asserted replacements so that they can be rehearsed on a COPY of the code tree and applied to the real tree when no job runs.
usage: python land_patch.py <project folder> [--apply]      without --apply: every anchor is checked and nothing is written. The `git rm`, the tags and the commits are in land.sh; this file edits the files that STAY:
  lane_past_data.py (+ its test): the S2-only `loader_fixture` (the one place that imports jpast), `jtiming.py` (+ `timed` from the E354 timing script), `token_time_ens.py` (uses jtiming.timed), the READMEs (rows of the removed files), run_all_tests.sh (the removed tests, + test_lane_suite), the cards' notes."""
import os
import re
import sys

root = os.path.abspath(sys.argv[1])
apply_ = "--apply" in sys.argv
changed = {}


import subprocess
READMES = ("training/README.md", "training/grid/README.md", "evaluation/README.md", "tools/README.md")
BLOBS = sys.argv[sys.argv.index("--head-blobs") + 1] if "--head-blobs" in sys.argv else None   # the READMEs are shared with other chats (uncommitted hunks in the working tree): the landing COMMITS HEAD + my edits only (commit_paths --blob), and this mode writes exactly those files to the folder


def read(rel):
    if rel not in changed:
        if BLOBS and rel in READMES:
            changed[rel] = subprocess.check_output(["git", "show", "HEAD:./" + rel], cwd=root).decode("utf-8")
        else:
            with open(os.path.join(root, rel), encoding="utf-8") as fh:
                changed[rel] = fh.read()
    return changed[rel]


def rep(rel, old, new, count=1):
    s = read(rel)
    assert s.count(old) == count, f"{rel}: anchor found {s.count(old)} times (want {count}): {old[:90]!r}"
    changed[rel] = s.replace(old, new)


def cut(rel, start, end, keep_end=True):
    """remove [start, end) (end kept)"""
    s = read(rel)
    assert s.count(start) == 1 and s.count(end) == 1, f"{rel}: markers {s.count(start)} / {s.count(end)}: {start[:60]!r} .. {end[:60]!r}"
    i, j = s.index(start), s.index(end)
    assert i < j, (rel, start[:40], end[:40])
    changed[rel] = s[:i] + s[j:]


def drop_lines(rel, pattern, count=None):
    s = read(rel)
    lines = s.split("\n")
    keep = [ln for ln in lines if not re.search(pattern, ln)]
    n = len(lines) - len(keep)
    assert n >= 1 and (count is None or n == count), f"{rel}: pattern {pattern!r} matched {n} lines (want {count})"
    changed[rel] = "\n".join(keep)


# ------------------------------------------------------------------------------------------------ lane_past_data.py: no jpast any more (the data / sequence layer stays for E384)
LPD = "training/data/lane_past_data.py"
cut(LPD, "LOADER_HOWS = (", "# ------------------------------------------------------------------------------------------------------------------------------ step 0: what the past could give")
cut(LPD, "    gd, bd = dict(n=30, median=0.5, p90=1.5, p99=3.0)", "    names = []\n")
rep(LPD, '    for name, n_default in (("fixture", 200), ("loader-fixture", 40)):', '    for name, n_default in (("fixture", 200),):')
rep(LPD, '    elif a.cmd in ("fixture", "loader-fixture"):\n        res = (fixture if a.cmd == "fixture" else loader_fixture)(a.split, a.n_pairs, a.seed, a.ident)', '    elif a.cmd == "fixture":\n        res = fixture(a.split, a.n_pairs, a.seed, a.ident)')
drop_lines(LPD, r"lane_past_data\.py loader-fixture --split val", 1)
rep(LPD, "The loader fixture repeats the token test through the TRAINING path (`PastData` items, `collate`, `jpast.make_past`); the identity pose, the pose entry of the previous item and a shifted previous frame must FAIL it.\n", "(The loader fixture through `jpast.make_past` left with the S2 block: archive tag archive/s2-s3-blocks-2026-11-18.)\n")
LPT = "training/grid/test_lane_past.py"
drop_lines(LPT, r'^    \("the loader fixture', 3)
rep(LPT, ',\n                loader=["loader-fixture", "--split", "val", "--n-pairs", "20"])[kind]', ')[kind]')
rep(LPT, ', ["loader-fixture", "--split", "val", "--n-pairs", "40"]])', '])')
rep(LPT, ", the fixture on 40 real VAL pairs, the loader fixture, the pinned numbers", ", the fixture on 40 real VAL pairs, the pinned numbers")
rep(LPT, "numbers), `loader` (the loader fixture). The pins are", "numbers). The pins are")

# ------------------------------------------------------------------------------------------------ jtiming.py: `timed` (from jrecur_time.py, which goes behind the E354 / jtok2 tag)
JT = "evaluation/jtiming.py"
s = read(JT)
assert "def timed(" not in s
anchor = "def load1():"
assert s.count(anchor) == 1
timed_src = '''def timed(fn, cuda):
    """-> (fn(), ms): two CUDA events with a sync in front and behind (the protocol of this module), or `perf_counter` on the CPU (moved here from `jrecur_time.py`, archive tag archive/jtok2-jrecur-2026-11-18)"""
    if not cuda:
        t = time.perf_counter()
        r = fn()
        return r, (time.perf_counter() - t) * 1000.0
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    e0.record()
    r = fn()
    e1.record()
    torch.cuda.synchronize()
    return r, e0.elapsed_time(e1)


'''
rep(JT, anchor, timed_src + anchor)
rep("evaluation/token_time_ens.py", "    import jrecur_time as JRT\n    import jtiming as JTG\n", "    import jtiming as JTG\n")
s = read("evaluation/token_time_ens.py")
assert s.count("JRT.timed") == 3, s.count("JRT.timed")
changed["evaluation/token_time_ens.py"] = s.replace("JRT.timed", "JTG.timed")

# ------------------------------------------------------------------------------------------------ the READMEs: the rows of the removed files
GONE = ("jpast.py", "jpast_train.py", "jpast_eval.py", "jmap.py", "jmap_train.py", "jmap_eval.py", "lane_map_data.py", "test_jpast.py", "test_jpast_train.py", "test_jmap.py", "test_jmap_train.py", "test_lane_map.py", "token_readout.py", "jtok2.py", "jtok2_train.py", "jtok2_eval.py",
        "test_jtok2.py", "jrecur.py", "jrecur_train.py", "jrecur_eval.py", "jrecur_time.py", "test_jrecur.py", "run_token_tests.sh")
for rel in ("training/README.md", "training/grid/README.md", "evaluation/README.md", "tools/README.md"):
    s = read(rel)
    out, dropped = [], []
    for ln in s.split("\n"):
        m = re.match(r"\| `(?:[a-z_/]*?/)?([A-Za-z0-9_]+\.(?:py|sh))`", ln)
        if m and m.group(1) in GONE:
            dropped.append(m.group(1))
            continue
        out.append(ln)
    changed[rel] = "\n".join(out)
    print(f"{rel}: rows dropped {dropped}")
# the tools README: keep_alive.sh and its test are removed (absorbed by jarm.py), the maintenance row loses their sentence and the run_token_tests mention
s = read("tools/README.md")
i = s.find("; `keep_alive.sh` (")
if i >= 0:
    j = s.find(" |", i)
    changed["tools/README.md"] = s[:i] + s[j:]
s = read("tools/README.md")
i = s.find("`run_token_tests.sh` (")
if i >= 0:
    j = s.find(") |", i)
    k = s.rfind("; ", 0, i)
    changed["tools/README.md"] = s[:k] + s[j + 1:] if k >= 0 else s

# the README mentions inside rows that stay
rep("training/README.md", " `loader-fixture` (the token test again through the TRAINING path: the loader's items and `jpast.make_past`, the pose entry against a fresh `rel_pose` of the pair; the identity pose, the previous item's entry and a shifted previous frame must FAIL),", "")
rep("training/README.md", "; `jrecur_train.py` (E354: the loop of `jends_train.py` over sequences of keyframes with the recursive feature state of `grid/jrecur.py`: `--seq K --bs B --state on|off`, the E341 loss over all B x K frames as one batch, one backward pass through the states, `state_contrib` / gate / clock lines in the log; tests `grid/test_jrecur.py`)", "")
if "(what `maintenance/keep_alive.sh` did)" in read("tools/README.md"):
    rep("tools/README.md", "(what `maintenance/keep_alive.sh` did)", "(what the old keep_alive.sh did; it is in git history)")

# ------------------------------------------------------------------------------------------------ run_all_tests.sh: the removed tests out, the lane-suite selftest in
RT = "tools/maintenance/run_all_tests.sh"
drop_lines(RT, r'(test_jtok2\.py|test_token_readout\.py)', 2)
rep(RT, 'run "pmap: the order-keeping fork-pool map of the CPU scorers" evaluation/pmap.py\n', 'run "pmap: the order-keeping fork-pool map of the CPU scorers" evaluation/pmap.py\nrun "lane suite of the evaluation (5 checks, about 9 minutes)" test_lane_suite.py\n')

# ------------------------------------------------------------------------------------------------ the cards that cite the removed code
NOTE = "\n**Code archived 2026-11-18 (D117 / D119, the lead's approval): the files this card names are no longer in main; `git checkout {tag} -- <path>` restores them, the tag {tag} holds the whole code state of this experiment.**\n"
for rel, tag in (("experiments/E368_s2-cross-attention-to-the-past/experiment.md", "archive/s2-s3-blocks-2026-11-18"), ("experiments/E374_s3-cross-attention-to-the-sd-map/experiment.md", "archive/s2-s3-blocks-2026-11-18"), ("experiments/E354_recursive-feature-state/experiment.md", "archive/jtok2-jrecur-2026-11-18"),
                 ("experiments/E359_node-type-head-per-source-softmax/experiment.md", "archive/jtok2-jrecur-2026-11-18")):
    s = read(rel)
    assert "Code archived 2026-11-18" not in s
    changed[rel] = s.rstrip("\n") + "\n" + NOTE.format(tag=tag)

if BLOBS:
    os.makedirs(BLOBS, exist_ok=True)
    for rel in READMES:
        if rel in changed:
            with open(os.path.join(BLOBS, rel.replace("/", "__")), "w", encoding="utf-8", newline=chr(10)) as fh:
                fh.write(changed[rel])
    print(f"head blobs: {[r for r in READMES if r in changed]} written to {BLOBS} (nothing else written)")
elif apply_:
    for rel, s in changed.items():
        with open(os.path.join(root, rel), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(s)
    os.makedirs(os.path.join(root, "temp", "landing"), exist_ok=True)
    with open(os.path.join(root, "temp", "landing", "changed.txt"), "w", encoding="utf-8") as fh:
        fh.write(chr(10).join(sorted(changed)) + chr(10))
    print(f"applied: {len(changed)} files (list in temp/landing/changed.txt)")
else:
    print(f"checked: every anchor found, {len(changed)} files would change (nothing written)")
