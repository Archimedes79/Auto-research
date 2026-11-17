cd /mnt/c/Auto-research/bev_fusion_lane_center
python3 - <<'EOF'
import ast, re, sys
files = ["training/data/lane_past_data.py", "training/grid/jpast.py", "training/train/jpast_train.py", "evaluation/jpast_eval.py", "training/grid/test_jpast.py", "training/grid/test_lane_past.py", "training/grid/test_jpast_train.py"]
for f in files:
    src = open(f, encoding="utf-8").read()
    tree = ast.parse(src)
    names = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names: names[(a.asname or a.name).split(".")[0]] = n.lineno
        elif isinstance(n, ast.ImportFrom):
            for a in n.names: names[a.asname or a.name] = n.lineno
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.value.id for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
    un = [k for k in names if k not in used]
    # unused function parameters / top-level defs never referenced
    defs = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    unref = [d for d in defs if len(re.findall(r"\b%s\b" % d, src)) < 2]
    print(f, "unused imports:", un, "defs referenced once:", unref)
EOF
