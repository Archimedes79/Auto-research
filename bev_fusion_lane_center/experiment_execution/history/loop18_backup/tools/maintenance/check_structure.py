"""Check that the project structure and its documents are still in step. Run after changes (plain python 3, no GPU):

    wsl python3 tools/maintenance/check_structure.py            # report
    wsl python3 tools/maintenance/check_structure.py --selftest # test the checker itself

Checks
  1. root: only the allowed files (README.md, CLAUDE.md, HANDOVER.md) and the known folders
  2. every folder that humans open has a README.md (see FOLDERS_WITH_README)
  3. paths named in backticks or links of the living documents exist (project path, path relative to the
     document, or a script name found in training/ evaluation/ tools/ reports/build/ temp/)
  4. every living document is listed here (LIVING) - history and generated files are not checked
Exit code 1 if anything is reported, so it can be used as a gate at the end of a loop.
"""
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ROOT_FILES = {"README.md", "CLAUDE.md", "HANDOVER.md", ".gitignore"}
ROOT_DIRS = {"data", "evaluation", "experiment_execution", "experiments", "literature", "models", "reports", "runs", "temp",
             "tools", "training", "_to_delete"}
FOLDERS_WITH_README = ["data", "evaluation", "evaluation/sparse_lanes", "evaluation/lane_grid", "experiment_execution", "experiments",
                       "literature", "models", "reports", "reports/build", "runs", "temp", "tools", "training", "_to_delete"]
# living documents: checked for dead paths (glob-free; folders are walked for *.md except SKIP)
LIVING = ["README.md", "CLAUDE.md", "HANDOVER.md", "data", "evaluation", "experiment_execution", "models", "reports", "runs", "temp",
          "tools", "training", "experiments/README.md", "literature/README.md", "literature/OVERVIEW.md"]
SKIP_PARTS = ("CLEANUP_LIST.md", "/history/", "/loop17_recovery/", "/logs/", "/runs/", "/plans/", "/_to_delete/", "/papers/", "/__pycache__/",
              "/literature_backup", "METRIC_RESPONSE.md", "STATE.md", "LEDGER.md", "TRIAGE_STATE.md", "STRUCTURE.md", "/source/",
              "register.md", "CATALOGUE.md")
CODE_DIRS = ["training", "evaluation", "tools", "reports", "temp", "experiment_execution", "models", "literature", "experiments"]
GENERIC = {"log.txt", "status.json", "metrics.json", "model.pt", "run.json", "run.log", "card.md", "results", "figures", "eval.json", "README.md", "card.json"}
SKIP_INDEX = ("data", "runs", "_to_delete")
TOKEN = re.compile(r"`([^`\n]+)`|\]\(([^)\s]+)\)")
PATHLIKE = re.compile(r"^[A-Za-z0-9_.\-/\\{}\[\]<>*~$:]+$")


def code_index():
    names = set(GENERIC)
    for d in CODE_DIRS:
        for dp, dn, fn in os.walk(os.path.join(ROOT, d)):
            dn[:] = [x for x in dn if x != "__pycache__"]
            names.update(fn)
            names.update(dn)
    return names


def living_files():
    out = []
    for item in LIVING:
        p = os.path.join(ROOT, item)
        if os.path.isfile(p):
            out.append(p)
        elif os.path.isdir(p):
            for dp, dn, fn in os.walk(p):
                dn[:] = [x for x in dn if x not in ("__pycache__",)]
                for f in fn:
                    if f.endswith(".md"):
                        q = os.path.join(dp, f).replace("\\", "/")
                        if not any(s in q for s in SKIP_PARTS):
                            out.append(q)
    return sorted(set(out))


def resolves(tok, md, names):
    t = tok.strip().rstrip(".,;:)").replace("\\", "/")
    t = t.split("#")[0].split(":")[0] if not t.startswith(("http", "C:")) else t
    if not t or t.startswith(("http", "mailto")) or any(c in t for c in "*<>{}[]$~") or "NN" in t or t.startswith(".") or t.startswith("/loop") or re.search(r"[0-9]/[0-9]", t) or t.startswith("C:") or t in (".", ".."):
        return True
    if re.fullmatch(r"[\w-]+/[\w.-]+", t) and t.split("/")[0] not in ROOT_DIRS and not re.search(r"\.(py|sh|md|csv|json|pdf|pptx)$", t):
        return True  # owner/repo of a GitHub repository, not a project path
    if "/" not in t and not re.search(r"\.(py|sh|ps1|md|csv|json|txt|yml|pdf|pptx|png)$", t):
        return True  # a word, a command or an id, not a path
    if t.startswith("/mnt/c/ML-Experiment/bev_fusion_lane_center/"):
        t = t[len("/mnt/c/ML-Experiment/bev_fusion_lane_center/"):]
    if t.startswith("C:/ML-Experiment/bev_fusion_lane_center/"):
        t = t[len("C:/ML-Experiment/bev_fusion_lane_center/"):]
    if t.startswith(("wsl", "python", "bash", "powershell", "git", "cd ")) or " " in t:
        return True
    for base in (ROOT, os.path.dirname(md), os.path.join(ROOT, "runs"), os.path.join(ROOT, "runs", "e17")):
        if os.path.exists(os.path.join(base, t)):
            return True
    if os.path.basename(t.rstrip("/")) in names and "/" not in t.strip("/"):
        return True
    return t.startswith(("~/", "../_to_delete", "_to_delete", "out/", "runs/e")) or t.startswith("..") and "_to_delete" in t


def check():
    problems = []
    for f in sorted(os.listdir(ROOT)):
        p = os.path.join(ROOT, f)
        if os.path.isfile(p) and f not in ROOT_FILES:
            problems.append(f"root: unexpected file {f}")
        if os.path.isdir(p) and f not in ROOT_DIRS and not f.startswith("."):
            problems.append(f"root: unexpected folder {f}/")
    for d in FOLDERS_WITH_README:
        if os.path.isdir(os.path.join(ROOT, d)) and not os.path.exists(os.path.join(ROOT, d, "README.md")):
            problems.append(f"{d}/: no README.md")
    names = code_index()
    for md in living_files():
        rel = os.path.relpath(md, ROOT).replace("\\", "/")
        text = open(md, encoding="utf-8", errors="ignore").read()
        seen = set()
        for m in TOKEN.finditer(text):
            tok = m.group(1) or m.group(2)
            line = text[text.rfind("\n", 0, m.start()) + 1:text.find("\n", m.end()) if text.find("\n", m.end()) >= 0 else len(text)]
            if "_to_delete" in line or "cleanup_20" in line:
                continue  # a line that says where something was moved to
            if tok in seen:
                continue
            seen.add(tok)
            if PATHLIKE.match(tok.strip()) and not resolves(tok, md, names):
                problems.append(f"{rel}: dead path `{tok}`")
    return problems


def selftest():
    assert resolves("README.md", os.path.join(ROOT, "README.md"), set())
    assert resolves("tools/maintenance/check_structure.py", os.path.join(ROOT, "README.md"), set())
    assert not resolves("tools/maintenance/does_not_exist.py", os.path.join(ROOT, "README.md"), set())
    assert resolves("check_structure.py", os.path.join(ROOT, "README.md"), code_index())
    assert not resolves("evaluation/nonexistent_folder/x.py", os.path.join(ROOT, "README.md"), code_index())
    print("check_structure selftest passed")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
        sys.exit(0)
    probs = check()
    print("\n".join(probs) if probs else "structure and documents are in step")
    print(f"{len(probs)} problem(s)")
    sys.exit(1 if probs else 0)
