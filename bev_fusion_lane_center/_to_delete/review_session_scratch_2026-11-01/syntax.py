import ast
import json
import sys

for f in sys.argv[1:]:
    if f.endswith(".json"):
        json.load(open(f, encoding="utf-8"))
    else:
        ast.parse(open(f, encoding="utf-8").read(), f)
    print("ok", f)
