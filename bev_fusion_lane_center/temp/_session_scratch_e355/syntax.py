"""throw-away: syntax check (ast.parse) of the E355 files in temp/e346, no import, no bytecode"""
import ast
import os

here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "e346")
for f in ("e355_lib.py", "e346_lib.py", "e346_arm.py"):
    ast.parse(open(os.path.join(here, f), encoding="utf-8").read(), f)
    print("syntax ok", f)
