import os, sys, json, traceback
sys.path.insert(0, "autoresearch/tools")
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml.json")
os.environ["AR_MACHINE_LOCK"] = p
import machine_lock as m
for lk in ({"owner": "x", "end": None, "exclusive": True}, {"end": "2999-01-01", "exclusive": True}, {"owner": "x", "end": 5, "exclusive": True},
           {"owner": "x", "end": "2999-01-01", "exclusive": "false"}):
    json.dump(lk, open(p, "w"))
    try:
        b = m.blocks("me")
        print(lk, "-> blocks:", bool(b), end="  ")
        print("run.py message:", f"waiting for the machine lock of {b['owner']}" if b else "-")
    except Exception as e:
        print(lk, "-> CRASH", type(e).__name__, e)
