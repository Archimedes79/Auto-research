"""control: run every test function of test_jmaps_r2 on its own and print PASS / FAIL (BFLC_STAGE=none: the unfixed code)"""
import sys, traceback, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_jmaps_r2 as T
for name in sys.argv[1:] or ("test_end_cells", "test_raster_2m", "test_targets", "test_loss", "test_keypoints", "test_pair_head", "test_glue", "test_flags_off"):
    try:
        getattr(T, name)()
        print("RESULT", name, "PASS")
    except BaseException as e:
        print("RESULT", name, "FAIL", type(e).__name__, str(e)[:150].replace("\n", " "))
