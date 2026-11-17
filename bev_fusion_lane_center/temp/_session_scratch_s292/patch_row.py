import io
p = "temp/_session_scratch_s292/readme_e343_new.md"
s = io.open(p, encoding="utf-8", newline="").read()
a = "at its start and, with `--expect-code-digest` (+ `--expect-code-card`), refuses before any dump when it is not the code of the rehearsal;"
b = "at its start; on TEST `--expect-code-card` is REQUIRED (the card of a full, passed, undisturbed VAL rehearsal of this system on the same device: the pin is ITS digest and its environment record, i.e. hashes of the small input files and the torch / numpy / shapely versions; `--expect-code-digest` is an optional extra check) and the run refuses before any dump when the code or the environment differ; modules a run loads that the digest does not cover are printed as `UNDIGESTED MODULE`; on TEST the line is fixed (`--boot 1000`, seeds 0 1, E321's dumps and card, 21 scenes), standalone `dump-b` / `dump-ends` are refused, and the control reproduction runs before any new dump;"
assert s.count(a) == 1
s = s.replace(a, b)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")
