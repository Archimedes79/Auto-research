"""Scratch: resolve the three token-conflict blocks in the integration worktree (run with cwd = the worktree)."""
import re


def resolve(path, chooser):
    s = open(path, encoding="utf-8", newline="").read()
    pat = re.compile(r"\n<<<<<<< ours\n(.*?)\n\|\|\|\|\|\|\| base\n(.*?)\n=======\n(.*?)\n>>>>>>> theirs\n", re.S)
    m = pat.search(s)
    assert m, path
    ours, base, theirs = m.group(1), m.group(2), m.group(3)
    s = s[:m.start()] + chooser(ours, base, theirs) + s[m.end():]
    assert "<<<<<<<" not in s and ">>>>>>>" not in s, path
    open(path, "w", encoding="utf-8", newline="").write(s)
    print("resolved", path)


# tests/README.md: the row of test_student_chain loses its section K ('all' is gone): ours; the text of the row continues right after the block
resolve("bev_fusion/tests/README.md", lambda o, b, t: o)
# test_grad_scaler.py: both import lines
resolve("bev_fusion/tests/test_grad_scaler.py", lambda o, b, t: "\n" + "from trainkit import amp as tk_amp, optim as tk_optim  # noqa: E402\nfrom toy_train import ToyLoss, ToyNet, same_tree  # noqa: E402\n")
# nuscenes_dataset.py: the comment names the folder at the repository root
resolve("bev_fusion/training/nuscenes_dataset.py", lambda o, b, t: "GtGenerator/ at the repository root")
