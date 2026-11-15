"""scratch: the modules a real VAL run (all --max 4, one seed) loads against the code digest"""
import argparse, json, os, sys, time
R = os.environ["BFLC_ROOT"]
sys.path.insert(0, os.path.join(R, "evaluation"))
import jtest_eval as J
t = time.time(); files = J.code_sha256(); print("digest files", len(files), f"cold {time.time() - t:.2f} s", flush=True)
t = time.time(); J.code_sha256(); print(f"warm {time.time() - t:.2f} s", flush=True)
a = argparse.Namespace(split="val", max=4, run_prefix="e324_s_s", seeds=[0], boot=20, no_stack=False, device="cpu", b_dump=None, control_ends_dump=None, control_card=None, test_id="", expect_code_digest=None, expect_code_card=None)
rc = J.cmd_all(a)
repo = os.path.realpath(os.path.join(R, ".."))
loaded = sorted({os.path.relpath(os.path.realpath(m.__file__), repo).replace(os.sep, "/") for m in list(sys.modules.values()) if getattr(m, "__file__", None) and os.path.realpath(m.__file__).startswith(repo + os.sep)
                 and "/temp/" not in os.path.realpath(m.__file__).replace(os.sep, "/")})
digest = set(files)
print("rc", rc, "loaded repo modules", len(loaded))
print("LOADED NOT IN DIGEST:", [x for x in loaded if x not in digest])
print("IN DIGEST NOT LOADED:", len([x for x in digest if x not in set(loaded)]), sorted(x for x in digest if x not in set(loaded))[:60])
json.dump(dict(loaded=loaded, digest=sorted(digest)), open(os.path.join(R, "temp", "e343", "_session_scratch_", "closure_probe.json"), "w"))
