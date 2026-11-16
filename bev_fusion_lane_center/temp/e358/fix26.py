import sys
sys.path.insert(0, "/mnt/c/Auto-research/bev_fusion_lane_center/temp/e358")
from edit_helper import apply
apply("/mnt/c/Auto-research/bev_fusion_lane_center/evaluation/jwire_verdict.py", [
("""    global ROOT
    old_root, io_out = ROOT, __import__("io")
    try:""", """    old_root = ROOT
    try:"""),
("""                with __import__("contextlib").redirect_stdout(io_out.StringIO()) as buf:""", """                with contextlib.redirect_stdout(io.StringIO()) as buf:"""),
("""def _selftest():
""", """def _selftest():
    global ROOT
    import contextlib
    import io
"""),
])
print("ok")
