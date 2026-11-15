import json
p = "/mnt/c/Auto-research/bev_fusion/temp/t26/base_run/config.json"
c = json.load(open(p)); c["name"] = "t26_base_run"; json.dump(c, open(p, "w"), indent=1); print("name ->", c["name"], "| steps", c["steps"], "| radar_window", c.get("radar_window"), "| cam_buffer", c["model"].get("cam_buffer"))
