"""throw-away: the two training/README.md rows of the bugfix branch that main had already moved on from, applied by hand to blob4_training_README_md (HEAD + the automatic part); prints the diff against HEAD"""
import difflib
import subprocess

REPO = "/mnt/c/Auto-research"
p = f"{REPO}/bev_fusion/temp/blob4_training_README_md"
text = open(p, encoding="utf-8").read()
misplaced = (" `python ../autoresearch/tools/archdraw.py check training/architecture.json` (run from the problem folder): every named file and symbol exists, every `.py` here is in a box or "
             "listed as not drawn, picture and map are current |")
row = [ln for ln in text.split("\n") if ln.startswith("| `interface.py` |")]
assert len(row) == 1 and row[0].endswith("| `python training/interface.py` |" + misplaced), "interface.py row is not as expected"
text = text.replace(row[0], row[0][:-len(misplaced)])
old = "| `extract.py` | differentiable extraction of the target architecture (its design record"
assert text.count(old) == 1
text = text.replace(old, "| `extract.py` | differentiable extraction of the five-layer architecture (its design record")
open(p, "w", encoding="utf-8", newline="").write(text)
head = subprocess.run(["git", "show", "HEAD:bev_fusion/training/README.md"], capture_output=True, cwd=REPO).stdout.decode("utf-8").split("\n")
for name, blob in (("bev_fusion/README.md", "blob4_README_md"), ("bev_fusion/training/README.md", "blob4_training_README_md")):
    h = subprocess.run(["git", "show", f"HEAD:{name}"], capture_output=True, cwd=REPO).stdout.decode("utf-8").split("\n")
    b = open(f"{REPO}/bev_fusion/temp/{blob}", encoding="utf-8").read().split("\n")
    d = [ln for ln in difflib.unified_diff(h, b, lineterm="", n=0) if ln.startswith(("+", "-")) and not ln.startswith(("+++", "---"))]
    print(f"== {name}: {len(d)} changed lines (HEAD -> blob)")
    for ln in d:
        print("  ", ln[:1], ln[1:110].replace("\n", " "), "...", ln[-70:])
