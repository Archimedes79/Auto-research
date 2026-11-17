import sys
sys.path.insert(0, "/mnt/c/Auto-research/bev_fusion_lane_center/temp/e358")
from edit_helper import apply

S = "/mnt/c/Auto-research/bev_fusion_lane_center/temp/_session_scratch_e358/"
apply(S + "mutate.py", [
    ("""def main(name):
    import jwire_data  # noqa: F401  (the real modules first: the mutant replaces only its own entry in sys.modules)""", """def main(name):
    sys.path.insert(0, os.path.join(ROOT, "temp", "_session_scratch_e358"))
    import test_staged as TS  # noqa: F401  (first: it imports the staged qends_eval / jends_eval / jwire_head before anything else can import the tree's copies)
    import jwire_data  # noqa: F401  (the real modules first: the mutant replaces only its own entry in sys.modules)"""),
    ("""    sys.path.insert(0, os.path.join(ROOT, "temp", "_session_scratch_e358"))
    import test_staged as TS  # noqa: F401  (imports the staged qends_eval / jends_eval / jwire_head: they come first on the path)
    real = sys.modules[name]""", """    real = sys.modules[name]"""),
])
