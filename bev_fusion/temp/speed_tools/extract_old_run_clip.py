import ast, subprocess
src = subprocess.run(["git", "show", "HEAD:bev_fusion/training/train.py"], capture_output=True, cwd="/mnt/c/Auto-research").stdout.decode("utf-8")
tree = ast.parse(src)
lines = src.splitlines()
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in ("run_clip", "_f"):
        seg = "\n".join(lines[node.lineno - 1:node.end_lineno])
        open(f"/mnt/c/Auto-research/bev_fusion/temp/speed_tools/old_{node.name}.txt", "w", encoding="utf-8").write(seg + "\n")
        print(node.name, node.lineno, node.end_lineno)
