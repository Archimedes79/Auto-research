"""E387 (S2b): the restore of the S2 files into main, with the S2b edits that were made and tested in the scratch tree temp/_session_scratch_s2b (run from the project folder, only when NO run_all_tests.sh runs and the `past_block` hook is in jtoken.py).
usage: python temp/_session_scratch_s2b/restore_s2b.py [--apply]     without --apply every anchor is checked and nothing is written
Files: the scratch versions of jpast.py, jpast_train.py, jpast_eval.py, test_jpast.py, test_jpast_train.py (FusedPastBlock, --block, time, their tests); token_readout.py and test_token_readout.py unchanged from the tag. README rows: the rows of the tag (edited for E387) are added to HEAD's four READMEs; the commit takes HEAD + the rows as --blob (other chats have uncommitted hunks in those files), the working files get the same rows. run_all_tests.sh gets the three test lines.
Writes temp/_session_scratch_s2b/blobs/<readout path with __> and temp/_session_scratch_s2b/restored.txt (the paths for commit_paths)."""
import os
import re
import subprocess
import sys

TAG = "archive/s2-s3-blocks-2026-11-18"
PROJ = "bev_fusion_lane_center"
SCR = "temp/_session_scratch_s2b"
apply_ = "--apply" in sys.argv
here = os.getcwd()
assert os.path.basename(here) == PROJ, here


def git_show(spec):
    return subprocess.check_output(["git", "show", spec], cwd=here).decode("utf-8")


def head(rel):
    return git_show(f"HEAD:./{rel}")


def tag(rel):
    return git_show(f"{TAG}:{PROJ}/{rel}")


if apply_:
    assert subprocess.run(["pgrep", "-f", "run_all_tests.sh"], capture_output=True).returncode != 0, "run_all_tests.sh is still running: never edit it (or the tree) under it"
    assert "past_block" in open("training/grid/jtoken.py", encoding="utf-8").read(), "the past_block hook is not in jtoken.py yet"

SCRATCH = {"training/grid/jpast.py": "training/grid/jpast.py", "training/grid/test_jpast.py": "training/grid/test_jpast.py", "training/train/jpast_train.py": "training/train/jpast_train.py",
           "training/grid/test_jpast_train.py": "training/grid/test_jpast_train.py", "evaluation/jpast_eval.py": "evaluation/jpast_eval.py"}
FROM_TAG = ["evaluation/token_readout.py", "evaluation/tests/test_token_readout.py"]

ROWS = {   # readme -> [(name of the row in the tag, the text added before the closing ' |', the name of an existing row of HEAD after which it goes)]
    "training/README.md": [("jpast_train.py", " **E387 (S2b): `--block {residual,fused}` (default residual = E368 bit for bit; a config without the key reads as residual).**", "jtoken_train.py")],
    "training/grid/README.md": [("jpast.py", " **E387 (S2b): `FusedPastBlock` = the cross-attention + a gated MLP fusion of the cross output and the input (30,932 parameters, zero-init MLP output = the head without the block), `attend_g` / `mix`; 61 mutants in `test_jpast.py`.**", "jtoken.py"),
                                ("test_jpast.py", "", "test_jtoken.py"), ("test_jpast_train.py", "", "test_jpast.py")],
    "evaluation/README.md": [("jpast_eval.py", " **E387: `time` = the head-alone cost of a run with and one without a block, QUIET verdict (after-sample with the 2.5 s wait), promoted from the scratch time_s2.py.**", "jtoken_eval.py"),
                             ("token_readout.py", "", "jpast_eval.py")],
}


def find_row(text, name):
    for ln in text.split("\n"):
        m = re.match(r"\| `(?:[a-z_/]*?/)?" + re.escape(name) + r"`", ln)
        if m:
            return ln
    return None


def with_rows(text, rows_spec, tag_text):
    lines = text.split("\n")
    for name, extra, after in rows_spec:
        if find_row(text, name) is not None:
            continue                                   # already there
        row = find_row(tag_text, name)
        assert row is not None, f"no row of {name} in the tag's README"
        if extra:
            assert row.endswith(" |"), row[-40:]
            row = row[:-2] + extra.rstrip() + " |"
        idx = [i for i, ln in enumerate(lines) if re.match(r"\| `(?:[a-z_/]*?/)?" + re.escape(after) + r"`", ln)]
        assert len(idx) == 1, f"anchor row {after}: {len(idx)}"
        lines.insert(idx[0] + 1, row)
        text = "\n".join(lines)
    return "\n".join(lines)


os.makedirs(os.path.join(SCR, "blobs"), exist_ok=True)
paths, blobs = [], []
for rel, src in SCRATCH.items():
    paths.append(rel)
    if apply_:
        with open(rel, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(open(os.path.join(SCR, src), encoding="utf-8").read())
for rel in FROM_TAG:
    paths.append(rel)
    if apply_:
        os.makedirs(os.path.dirname(rel) or ".", exist_ok=True)
        with open(rel, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(tag(rel))
for rel, spec in ROWS.items():
    tag_text = tag(rel)
    new_head = with_rows(head(rel), spec, tag_text)
    blob = os.path.join(SCR, "blobs", rel.replace("/", "__"))
    blobs.append((rel, blob))
    with open(blob, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(new_head)
    if apply_:
        work = open(rel, encoding="utf-8").read()
        with open(rel, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(with_rows(work, spec, tag_text))
# run_all_tests.sh: the three test lines after the token-cache line
RT = "tools/maintenance/run_all_tests.sh"
rt = open(RT, encoding="utf-8").read()
anchor = 'run "E365 token cache: build / read / selftest" grid/test_lane_tokcache.py\n'
assert rt.count(anchor) == 1
add = ('run "E368 / E387 cross-attention blocks to the past keyframe (residual and fused; hand values, loop references, 61 mutants)" grid/test_jpast.py\n'
       'run "E368 / E387 past trainer and stream read-out (CPU, stub and real head)" grid/test_jpast_train.py\n'
       'run "read-out helpers of the token heads (every-3 sample, existence AP, cell choice)" test_token_readout.py\n')
if "grid/test_jpast.py" not in rt:
    rt2 = rt.replace(anchor, anchor + add)
    paths.append(RT)
    if apply_:
        with open(RT, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(rt2)
with open(os.path.join(SCR, "restored.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(paths) + "\n" + "\n".join(f"{r}={b}" for r, b in blobs) + "\n")
print(("applied" if apply_ else "checked (nothing written)"), len(paths), "files,", len(blobs), "README blobs:", [r for r, _ in blobs])
