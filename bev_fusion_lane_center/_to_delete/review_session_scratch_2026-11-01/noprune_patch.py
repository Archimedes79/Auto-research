p = "autoresearch/tools/provenance.py"
s = open(p, encoding="utf-8").read()
old = '            git(repo, "worktree", "prune")\n'
assert s.count(old) == 1
s = s.replace(old, "")
old2 = "(the half-made worktree was removed; check the leftovers in that folder)"
assert s.count(old2) == 1
s = s.replace(old2, "(the half-made worktree was removed; check the leftovers in that folder; never run `git worktree prune` from WSL: it drops the registration of worktrees with Windows paths)")
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
