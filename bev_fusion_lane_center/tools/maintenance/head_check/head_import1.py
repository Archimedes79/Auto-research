"""The clean-copy check (tools/maintenance/head_check/head_check.sh), part 3: import ONE module file for real (its own folder first on sys.path and as cwd, as `_py14.sh` runs it) and print one line:
'IMPORT OK <file>' or 'IMPORT FAIL <file>: <error> (<where>)'. The driver runs it under `timeout 60` per file (a TIMEOUT = the module runs work at import).
usage: _py14.sh <copy>/bev_fusion_lane_center/temp/head_import1.py <module file>"""
import importlib.util
import os
import re
import sys
import traceback

path = os.path.abspath(sys.argv[1])
C = os.environ["BFLC_ROOT"]
rel = os.path.relpath(path, C)
sys.argv = [path]
sys.path.insert(0, os.path.dirname(path))
os.chdir(os.path.dirname(path))
try:
    d, parts = os.path.dirname(path), [os.path.splitext(os.path.basename(path))[0]]
    while os.path.isfile(os.path.join(d, "__init__.py")):                    # a module of a package: imported by its dotted name from above the package (as its users do)
        parts.insert(0, os.path.basename(d))
        d = os.path.dirname(d)
    if len(parts) > 1:
        sys.path.insert(0, d)
        importlib.import_module(".".join(p for p in parts if p != "__init__"))
    else:
        spec = importlib.util.spec_from_file_location("headchk_" + re.sub(r"\W", "_", rel), path)
        spec.loader.exec_module(importlib.util.module_from_spec(spec))
    print(f"IMPORT OK {rel}")
except BaseException as e:
    tb = traceback.extract_tb(e.__traceback__)
    msg = str(e).splitlines()[0][:160] if str(e) else ""
    print(f"IMPORT FAIL {rel}: {type(e).__name__}: {msg} ({os.path.relpath(tb[-1].filename, C) if tb else ''}:{tb[-1].lineno if tb else ''})")
