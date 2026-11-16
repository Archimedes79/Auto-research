import json, os
R = os.environ["BFLC_ROOT"]
d = json.load(open(os.path.join(R, "temp/e345/e2e/sweep.json")))
d["run"] = "e312_p_s1"
os.makedirs(os.path.join(R, "temp/e345/e2e_s1"), exist_ok=True)
json.dump(d, open(os.path.join(R, "temp/e345/e2e_s1/sweep.json"), "w"))
print("ok", len(d["passes"]))
