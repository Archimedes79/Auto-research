import json
c = json.load(open("temp/t22/smoke/smoke_t22_B_s0.json"))
c["name"] = "smoke_t22_B_s0b"
json.dump(c, open("temp/t22/smoke/smoke_t22_B_s0b.json", "w"), indent=1)
print(c["name"], c["steps"], c["log_every"], c["eval_every"], c["eval_batches"])
