"""Read-only: for each given file, path-like tokens (dir/.../name.ext with ext py|md|json|sh|csv|txt) that resolve to nothing relative to
bev_fusion/, the repo root or the file's folder, and are not temp/runs/data/results run artefacts (reported separately as 'runtime')."""
import os, re, sys

REPO = "/mnt/c/Auto-research"
BF = REPO + "/bev_fusion"
PAT = re.compile(r"(?<![\w./-])((?:\.\./)*(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.(?:py|md|json|sh|csv|txt|ps1))\b")
RUNTIME = ("temp/", "runs/", "data/cache", "results/", "records/runs", "eval/", "prototype/", "~/", "/home", "/mnt", "D:", "models/pretrained", "literature/papers/")
for f in sys.argv[1:]:
    p = os.path.join(REPO, f)
    if not os.path.exists(p):
        continue
    txt = open(p, encoding="utf-8", errors="replace").read()
    seen = set()
    for ln, line in enumerate(txt.splitlines(), 1):
        for m in PAT.finditer(line):
            tok = m.group(1)
            if tok in seen:
                continue
            seen.add(tok)
            if any(r in tok for r in RUNTIME) or "<" in tok or "*" in tok:
                continue
            cands = [os.path.join(BF, tok), os.path.join(REPO, tok), os.path.join(os.path.dirname(p), tok), os.path.join(REPO, "autoresearch", tok),
                     os.path.join(BF, "training", tok), os.path.join(BF, "evaluation", tok), os.path.join(REPO, tok.lstrip("./"))]
            if not any(os.path.exists(c) for c in cands):
                print(f"{f}:{ln}: {tok}")
