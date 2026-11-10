p = "tests/test_evaluation_api.py"
s = open(p, encoding="utf-8").read()
old = '''        want = next(fr["scores"] for fr in store.frames if fr["label"] == f"{scene}/{index:03d}")'''
new = '''        want = {k: v for k, v in next(fr["scores"] for fr in store.frames if fr["label"] == f"{scene}/{index:03d}").items() if not k.startswith("loss.")}    # the Runner reads no ground truth: no val loss'''
assert s.count(old) == 1
open(p, "w", encoding="utf-8").write(s.replace(old, new))
print("ok")
