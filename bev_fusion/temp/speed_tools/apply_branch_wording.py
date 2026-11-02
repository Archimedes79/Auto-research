"""throw-away: the one commit of origin/bugfix/codebase-review-2026-11-01 (README wording) applied by hand to HEAD's text. For every line it changed (base -> theirs, found by line matching) the
contiguous differing block is replaced in HEAD's version of the same line, only if HEAD's line still has the same start and end as the base line and the same old middle; otherwise it is reported.
Writes temp/blob4_<name> (HEAD + the applied changes), prints what was applied and what was not."""
import difflib
import subprocess

REPO = "/mnt/c/Auto-research"
BR = "origin/bugfix/codebase-review-2026-11-01"
git = lambda *a: subprocess.run(["git", *a], capture_output=True, cwd=REPO).stdout.decode("utf-8")      # noqa: E731
MB = git("merge-base", "origin/main", BR).strip()
for path, blob in (("bev_fusion/README.md", "blob4_README_md"), ("bev_fusion/training/README.md", "blob4_training_README_md")):
    base, theirs, head = (git("show", f"{r}:{path}").split("\n") for r in (MB, BR, "HEAD"))
    out = list(head)
    ops = [op for op in difflib.SequenceMatcher(None, base, theirs, autojunk=False).get_opcodes() if op[0] != "equal"]
    applied = failed = 0
    for tag, i1, i2, j1, j2 in ops:
        if tag != "replace" or (i2 - i1) != (j2 - j1):
            print(f"  {path}: opcode {tag} {i1}:{i2} -> {j1}:{j2} is not a 1:1 line change: NOT applied")
            failed += 1
            continue
        for b, t in zip(base[i1:i2], theirs[j1:j2]):
            p = 0
            while p < min(len(b), len(t)) and b[p] == t[p]:
                p += 1
            s = 0
            while s < min(len(b), len(t)) - p and b[len(b) - 1 - s] == t[len(t) - 1 - s]:
                s += 1
            old, new = b[p:len(b) - s], t[p:len(t) - s]
            cand = [k for k, h in enumerate(out) if h.startswith(b[:p][:80]) and h.endswith(b[len(b) - s:][-60:] if s else "")]
            hit = [k for k in cand if out[k][:p] == b[:p] and (out[k][len(out[k]) - s:] == b[len(b) - s:] if s else True) and out[k][p:len(out[k]) - s] == old]
            if len(hit) == 1:
                out[hit[0]] = out[hit[0]][:p] + new + out[hit[0]][len(out[hit[0]]) - s:] if s else out[hit[0]][:p] + new
                applied += 1
                print(f"  {path}: applied: [{old[:60]!r}...] -> [{new[:60]!r}...]")
            else:
                failed += 1
                print(f"  {path}: NOT applied (HEAD's line differs from the base in this block; candidates {len(cand)}): [{old[:70]!r}...]")
    open(f"{REPO}/bev_fusion/temp/{blob}", "w", encoding="utf-8", newline="").write("\n".join(out))
    print(f"{path}: {applied} applied, {failed} not")
