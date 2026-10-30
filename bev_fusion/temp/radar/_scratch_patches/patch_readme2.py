p = "training/README.md"
s = open(p, encoding="utf-8").read()
row = ("| `radar_encoder.py` | radar block that keeps the information (plan.md 3.4): delta-coded slot raster (K = 3 exact slots + remainder mean + count; per slot valid, dr = r - \|c - s\|, dpsi = psi - bearing(c - s), Doppler code, rcs, age, sub-cell offset, ray origin s; position = cell index + offset, r and psi decoded exactly from cell + origin + residual) -> two 3 x 3 convs, LeakyReLU, C >= inputs, pass-through start; `aux_loss` = tolerance-weighted reconstruction as a SIDE loss with `aux_weight(step)` decaying to a floor (the task losses are the goal), `kept_fraction` = measurement of what the features still hold (no gradient), `forward(..., with_state=True)` = the lossless raster as explicit stream; `rasterize_delta_slots` / `unpack` / `decode_slots`, `new_decoder` for probing frozen features | exact round trip of eight fields (two frames, several sensor poses), residual bound, overflow + remainder mean, Doppler wrap, supplied ray origin, module shapes, side-loss gradients, pass-through start has the lower side loss than a default start |\n")
anchor = "| `splat.py` |"
s = s.replace(anchor, row + anchor, 1)
open(p, "w", encoding="utf-8").write(s)
print("ok")
