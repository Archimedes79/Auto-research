import json, sys
sys.path.insert(0, "training")
import splits
d = splits.resolve(json.load(open("training/configs/t26_one.json")))
d = {k: v for k, v in d.items() if k != "_split"}
json.dump(d, open("temp/t26/composed_t26.json", "w"))
