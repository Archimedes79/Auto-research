import sys
sys.path.insert(0, "/mnt/c/Auto-research/bev_fusion_lane_center/temp/e358")
from edit_helper import apply
apply("/mnt/c/Auto-research/bev_fusion_lane_center/temp/_session_scratch_e358/test_staged.py", [
("""import jends_eval as JE  # noqa: E402
""", """import jends_eval as JE  # noqa: E402
import qgraph_eval  # noqa: E402,F401  (`cmd_score` imports them: before a test points BFLC_ROOT at a temp folder, which their own path set-up reads)
import qgraph_readout  # noqa: E402,F401
"""),
])
print("ok")
