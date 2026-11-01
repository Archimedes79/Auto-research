p = "autoresearch/tools/provenance.py"
s = open(p, encoding="utf-8").read()
def sub(old, new):
    global s
    assert s.count(old) == 1, (old[:60], s.count(old))
    s = s.replace(old, new)
sub('def git(repo, *args):', 'def git(repo, *args, timeout=60):')
sub('capture_output=True, text=True, timeout=60)\n    except (OSError, subprocess.SubprocessError):\n        return None', 'capture_output=True, text=True, timeout=timeout)\n    except (OSError, subprocess.SubprocessError):\n        return None')
sub('if git(repo, "worktree", "add", "--detach", dest, tag) is None:', 'if git(repo, "worktree", "add", "--detach", dest, tag, timeout=3600) is None:   # a checkout on the Windows disk from WSL is slow')
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
