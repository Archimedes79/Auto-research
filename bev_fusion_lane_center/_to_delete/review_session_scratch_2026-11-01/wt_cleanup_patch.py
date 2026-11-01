p = "autoresearch/tools/provenance.py"
s = open(p, encoding="utf-8").read()
old = '''            sys.exit(f"git worktree add {dest} {tag} failed")'''
new = '''            git(repo, "worktree", "unlock", dest)                      # a failed or timed-out checkout leaves a locked, half-made worktree: take it away again
            git(repo, "worktree", "remove", "--force", dest)
            git(repo, "worktree", "prune")
            sys.exit(f"git worktree add {dest} {tag} failed (the half-made worktree was removed; check the leftovers in that folder)")'''
assert s.count(old) == 1
open(p, "w", encoding="utf-8", newline="\n").write(s.replace(old, new))
print("ok")
