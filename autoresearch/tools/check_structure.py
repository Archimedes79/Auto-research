"""Check that the problem folder's structure and its documents are still in step (gate G4). Rules come from project.json "structure".

    python autoresearch/tools/check_structure.py            # report, exit 1 if anything is found
    python autoresearch/tools/check_structure.py --selftest

Checks
  1. root of the problem folder: only the allowed files (root_files) and folders (root_dirs)
  2. every folder humans open has a README.md (folders_with_readme)
  3. paths named in backticks or links of the living documents (living, minus skip_parts) exist: as a path from the problem folder, from the
     document's folder, from one of extra_bases, or as a file / folder name found in index_dirs
"""
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import project as P  # noqa: E402

GENERIC = {"log.txt", "status.json", "metrics.json", "model.pt", "run.json", "run.log", "card.md", "results", "figures", "eval.json", "README.md", "card.json"}
TOKEN = re.compile(r"`([^`\n]+)`|\]\(([^)\s]+)\)")
PATHLIKE = re.compile(r"^[A-Za-z0-9_.\-/\\{}\[\]<>*~$:]+$")
EXT = r"\.(py|sh|ps1|md|csv|json|txt|yml|pdf|pptx|png)$"


def rules():
    s = P.cfg()["structure"]
    here = P.root().replace("\\", "/")  # absolute spellings of the problem folder (WSL and Windows) are stripped too, wherever the repository lives
    here_win = re.sub(r"^/mnt/([a-z])/", lambda m: m.group(1).upper() + ":/", here)
    return dict(root_files=set(s.get("root_files", [])) | {".gitignore"}, root_dirs=set(s.get("root_dirs", [])), readme=s.get("folders_with_readme", []),
                living=s.get("living", []), skip=tuple(s.get("skip_parts", [])), index=s.get("index_dirs", []), bases=s.get("extra_bases", []),
                strip=list(s.get("strip_prefixes", [])) + [here + "/", here_win + "/"])


def code_index(root, R):
    names = set(GENERIC)
    for top in [os.path.join(root, d) for d in R["index"]] + [P.AR]:  # the problem's folders and autoresearch itself
        for dp, dn, fn in os.walk(top):
            dn[:] = [x for x in dn if x != "__pycache__"]
            names.update(fn)
            names.update(dn)
    return names


def living_files(root, R):
    out = []
    for item in R["living"]:
        p = os.path.join(root, item)
        if os.path.isfile(p):
            out.append(p)
        elif os.path.isdir(p):
            for dp, dn, fn in os.walk(p):
                dn[:] = [x for x in dn if x != "__pycache__"]
                for f in fn:
                    q = os.path.join(dp, f).replace("\\", "/")
                    rel = "/" + os.path.relpath(q, root).replace("\\", "/")        # the skip parts name folders INSIDE the problem folder, not above it
                    if f.endswith(".md") and not any(s in rel for s in R["skip"]):
                        out.append(q)
    return sorted(set(out))


def resolves(tok, md, names, root, R):
    t = tok.strip().rstrip(".,;:)").replace("\\", "/")
    t = t.split("#")[0].split(":")[0] if not t.startswith(("http", "C:")) else t
    relative = t.startswith(("./", "../"))                  # resolved against the document and the problem folder below; other dot words (.gitignore, ...) are no path
    if not t or t.startswith(("http", "mailto", "/loop", "C:")) or (t.startswith(".") and not relative) or any(c in t for c in "*<>{}[]$~") or "NN" in t or re.search(r"[0-9]/[0-9]", t):
        return True
    if re.fullmatch(r"[\w-]+/[\w.-]+", t) and t.split("/")[0] not in R["root_dirs"] and not re.search(r"\.(py|sh|md|csv|json|pdf|pptx)$", t):
        return True  # owner/repo of a GitHub repository
    if "/" not in t and not re.search(EXT, t):
        return True  # a word, a command or an id
    for pre in R["strip"]:
        if t.startswith(pre):
            t = t[len(pre):]
    if t.startswith(("wsl", "python", "bash", "powershell", "git")):
        return True
    for base in [root, os.path.dirname(md), os.path.dirname(root)] + [os.path.join(root, b) for b in R["bases"]]:
        if os.path.exists(os.path.join(base, t)):
            return True
    if os.path.basename(t.rstrip("/")) in names and "/" not in t.strip("/"):
        return True
    return t.startswith(("~/", "out/", "runs/e"))


def check(root=None):
    root = root or P.root()
    R = rules()
    problems = []
    for f in sorted(os.listdir(root)):
        p = os.path.join(root, f)
        if os.path.isfile(p) and f not in R["root_files"]:
            problems.append(f"root: unexpected file {f}")
        if os.path.isdir(p) and f not in R["root_dirs"] and not f.startswith("."):
            problems.append(f"root: unexpected folder {f}/")
    for d in R["readme"]:
        if os.path.isdir(os.path.join(root, d)) and not os.path.exists(os.path.join(root, d, "README.md")):
            problems.append(f"{d}/: no README.md")
    names = code_index(root, R)
    for md in living_files(root, R):
        rel = os.path.relpath(md, root).replace("\\", "/")
        text = open(md, encoding="utf-8", errors="ignore").read()
        seen = set()
        for m in TOKEN.finditer(text):
            tok = m.group(1) or m.group(2)
            e = text.find("\n", m.end())
            line = text[text.rfind("\n", 0, m.start()) + 1:e if e >= 0 else len(text)]
            if "_to_delete" in line or "cleanup_20" in line or tok in seen:
                continue  # a line that says where something was moved to
            seen.add(tok)
            if PATHLIKE.match(tok.strip()) and not resolves(tok, md, names, root, R):
                problems.append(f"{rel}: dead path `{tok}`")
    return problems


def selftest():
    root, R = P.root(), rules()
    md = os.path.join(root, "README.md")
    names = code_index(root, R)
    assert resolves("README.md", md, set(), root, R)
    assert not resolves("tools/does_not_exist_anywhere.py", md, set(), root, R)
    assert not resolves("evaluation/nonexistent_folder/x.py", md, names, root, R)
    assert resolves("autoresearch/tools/run.py", md, set(), root, R)
    assert resolves("../autoresearch/tools/run.py", md, set(), root, R) and resolves(".gitignore", md, set(), root, R)
    assert not resolves("../autoresearch/tools/does_not_exist.py", md, set(), root, R) and not resolves("./does_not_exist.py", md, set(), root, R), "a relative path is looked up"
    import tempfile
    tmp = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp, "runs", "proj", "docs", "history"))                   # a checkout below a folder called runs
    for f in ("docs/a.md", "docs/history/old.md"):
        open(os.path.join(tmp, "runs", "proj", f), "w").write("x\n")
    got = living_files(os.path.join(tmp, "runs", "proj"), dict(living=["docs"], skip=("/runs/", "/history/")))
    assert [os.path.basename(x) for x in got] == ["a.md"], "skip parts are matched inside the problem folder, not on the path above it"
    shutil.rmtree(tmp)
    print("check_structure selftest passed")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
        sys.exit(0)
    probs = check()
    print("\n".join(probs) if probs else "structure and documents are in step")
    print(f"{len(probs)} problem(s)")
    sys.exit(1 if probs else 0)
