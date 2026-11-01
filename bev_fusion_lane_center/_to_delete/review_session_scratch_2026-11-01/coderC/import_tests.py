"""imports the tests / temp scripts that use the moved modules (their __main__ guards keep the work from running); run from bev_fusion"""
import importlib.util
import sys

for path in ("tests/test_s1_pipeline.py", "tests/test_camera_only.py", "tests/test_radar_only.py", "temp/timing/time_s1_chain.py"):
    saved = list(sys.path)
    spec = importlib.util.spec_from_file_location("m_" + path.replace("/", "_")[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.path[:] = saved
    print("import ok:", path)
