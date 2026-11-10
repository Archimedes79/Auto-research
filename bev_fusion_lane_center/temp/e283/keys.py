import json, sys
c = json.load(open(sys.argv[1]))
def walk(d, pre="", depth=0):
    for k, v in d.items():
        if isinstance(v, dict) and depth < 3:
            print(" " * depth + str(k) + ":" , list(v.keys())[:40] if not any(isinstance(x, dict) for x in v.values()) else "{...}")
            walk(v, pre + k + ".", depth + 1)
walk(c)
