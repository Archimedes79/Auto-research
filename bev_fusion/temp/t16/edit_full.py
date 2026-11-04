import json
p = "bev_fusion/training/configs/t16_full.json"
c = json.load(open(p, encoding="utf-8"))
m = c["model"]
for k in ("rad_temporal", "rad_static_gate"):
    m.pop(k, None)                                    # the radar memory leaves the large run: its static map is 1.0 everywhere and its velocity head gets no gradient (probe, S4)
c["grad_clip_split"] = ["motion_block"]                # the motion block's gradient is 30 x the others': clipped on its own
c["_about"] = (c["_about"].replace("plus every block that is built and has a passing gate: the radar delta-slot encoder R (c_rad 64, side loss radar_aux), the radar memory with the Doppler static gate, the motion block M",
                                   "plus the blocks that are built, have a passing gate and survived the small-set bring-up: the radar delta-slot encoder R (c_rad 64, side loss radar_aux), the motion block M")
               + " The radar memory with the Doppler static gate (rad_temporal window_memory + rad_static_gate) is OFF here: in the small-set probe its learned static map was exactly 1.0 everywhere and its velocity head got no gradient (card section 6); it returns as its own arm. grad_clip_split clips the motion block's gradient (30 x the others') on its own.")
json.dump(c, open(p, "w", encoding="utf-8"), indent=1)
open(p, "a", encoding="utf-8").write("\n")
print(c["model"], c["grad_clip_split"])
