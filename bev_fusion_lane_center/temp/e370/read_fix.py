import json, glob
for S in (0, 1):
    for f in sorted(glob.glob(f"runs/e370_a_s{S}/token_fixture*.json")):
        d = json.load(open(f))
        print(S, f.split("/")[-1])
        print("  ", json.dumps({k: v for k, v in d.items() if not isinstance(v, (list, dict))})[:600])
        for k, v in d.items():
            if isinstance(v, dict):
                print("  ", k, json.dumps(v)[:500])
