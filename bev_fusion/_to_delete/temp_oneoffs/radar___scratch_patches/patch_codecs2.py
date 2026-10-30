p = "tests/radar_codecs.py"
s = open(p, encoding="utf-8").read()
# helper for the failure share per field
s = s.replace('''def level_d(steps):''', '''def fail_share(e):
    """share of all detections that violate the tolerance of each field (invalid slots count as violations)."""
    return ", ".join(f"{100 * (e[k] > t).float().mean().item():.0f}" for k, t in rt.TOL.items())


def level_d(steps):''', 1)
s = s.replace('''print("  codec / variant     | channels in | C   | held-out | train | gap  | held-out median: " + ", ".join(rt.TOL))''',
              '''print("  codec / variant     | channels in | C   | held-out | train | gap  | held-out median: " + ", ".join(rt.TOL) + " | violating % per field")''')
s = s.replace('''| {med}", flush=True)''', '''| {med} | {fail_share(e)}", flush=True)''')
open(p, "w", encoding="utf-8").write(s)
print(s.count("fail_share"))
