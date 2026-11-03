"""adds the static_weight words to the radar_encoder row of training/README.md in (1) the HEAD version (for the commit) and (2) the working copy."""
import subprocess

old_a = "`rasterize_delta_slots` / `unpack` / `decode_slots`, `new_decoder` for probing frozen features |"
new_a = ("`rasterize_delta_slots` / `unpack` / `decode_slots`, `new_decoder` for probing frozen features, `static_weight(state)` = a parameter-free "
         "static prior of every cell from the Doppler of the raster (exp(-(vr / sigma)^2), count-weighted over the slots; 1.0 for an empty cell; "
         "necessary, not sufficient: a crossing object reads about 0) for a radar memory, to be multiplied with the learned static map |")
old_b = "side-loss gradients, pass-through start has the lower side loss than a default start |"
new_b = ("side-loss gradients, pass-through start has the lower side loss than a default start, Doppler static weight (static 1, moving 0, "
         "mixed cell, overflow cell with a moving remainder, empty cell 1.0, frames do not mix) |")

head = subprocess.run(["git", "-C", "..", "show", "HEAD:bev_fusion/training/README.md"], capture_output=True, check=True).stdout.decode("utf-8")
for name, text in (("temp/radar/readme_mine.md", head), ("training/README.md", open("training/README.md", encoding="utf-8").read())):
    assert text.count(old_a) == 1 and text.count(old_b) == 1, (name, text.count(old_a), text.count(old_b))
    text = text.replace(old_a, new_a, 1).replace(old_b, new_b, 1)
    open(name, "w", encoding="utf-8", newline="").write(text)
    print("written", name)
