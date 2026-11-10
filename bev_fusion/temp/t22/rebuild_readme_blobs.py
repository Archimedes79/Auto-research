"""scratch: rebuilds the two README copies for the commit from the CURRENT HEAD plus only my edits (the two rounds of readme_edit.py / readme_edit2.py, read from their EDITS literals),
so that rows another chat committed since the first build are kept. Writes temp/t22/blob_tests_README.md and blob_training_README.md. Run from the repository root."""
import ast
import subprocess


def edits_of(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "EDITS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise SystemExit(f"no EDITS in {path}")


e1 = edits_of("bev_fusion/temp/t22/readme_edit.py")          # {path: [(old, new)]}
e2 = edits_of("bev_fusion/temp/t22/readme_edit2.py")         # {path: (blob, [(old, new)])}
for path, edits in e1.items():
    head = subprocess.run(["git", "show", "HEAD:" + path], capture_output=True, check=True).stdout.decode("utf-8")
    blob, edits2 = e2[path]
    for old, new in list(edits) + list(edits2):
        assert head.count(old) == 1, (path, old[:60], head.count(old))
        head = head.replace(old, new)
    open(blob, "w", encoding="utf-8", newline="").write(head)
    print("rebuilt", blob, "from HEAD", subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip())
