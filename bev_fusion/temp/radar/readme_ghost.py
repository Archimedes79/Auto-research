"""adds the test_radar_memory_ghost.py row to tests/README.md: in the HEAD version (for the commit, so other chats' edits stay out) and in the working copy."""
import subprocess

row = ("| `test_radar_memory_ghost.py` | does the Doppler write mask of the radar memory (`rad_static_gate`, `precompute.radar_static`) keep a mover out of "
       "`TemporalFusion`'s world memory? synthetic world of the temporal_fusion self-test (static pattern, radar at the origin, ego standing still, dense radar, "
       "a 6 m/s blob), memory read behind the blob vs the static world: no gate / radial mover with the gate / crossing mover with the gate; CPU, seconds | "
       "no gate > 0.1, radial < 0.02, crossing between 3 x radial and the no-gate value | passes (2026-11-02): ghost 0.129 / 0.002 / 0.051 - the Doppler of a "
       "crossing object vanishes at the crossing point (card t19_radar_memory) |\n")
anchor = "| `test_object_head.py`"
head = subprocess.run(["git", "-C", "..", "show", "HEAD:bev_fusion/tests/README.md"], capture_output=True, check=True).stdout.decode("utf-8")
for name, text in (("temp/radar/tests_readme_mine.md", head), ("tests/README.md", open("tests/README.md", encoding="utf-8").read())):
    assert text.count(anchor) == 1 and "test_radar_memory_ghost" not in text, name
    open(name, "w", encoding="utf-8", newline="").write(text.replace(anchor, row + anchor, 1))
    print("written", name)
