import os, sys, time
pp = os.environ.get("PYTHONPATH", "").split(":")
print("PYTHONPATH entries", len(pp))
print([p.replace(os.environ["BFLC_ROOT"], "") for p in pp[:8]], "...")
