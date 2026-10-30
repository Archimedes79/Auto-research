import os, runpy, sys
sys.path.insert(0, "training")
os.chdir(".")
sys.argv = ["model.py"]
runpy.run_path("temp/checks/model_mine/model.py", run_name="__main__")
