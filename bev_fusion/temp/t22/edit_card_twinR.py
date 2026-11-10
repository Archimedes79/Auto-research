"""scratch: insert the 'Seed twin and R arm' bullet (temp/t22/twinR_card_bullet.txt) after the Stage B2 result bullet of the t22 card (and fix the 'still open' sentence of the status line);
asserts every anchor once; CRLF/LF of the file is kept."""
P = "experiments/t22_student_encoder/experiment.md"
raw = open(P, "rb").read().decode("utf-8")
crlf = "\r\n" in raw
text = raw.replace("\r\n", "\n")
lines = text.split("\n")
bullet = open("temp/t22/twinR_card_bullet.txt", encoding="utf-8").read().rstrip("\n")

assert not any(l.startswith("  - **Seed twin and R arm (") for l in lines), "already inserted"
bi = [i for i, l in enumerate(lines) if l.startswith("  - **Stage B2 result (")]
assert len(bi) == 1, bi
lines[bi[0] + 1:bi[0] + 1] = bullet.split("\n")

si = [i for i, l in enumerate(lines) if l.startswith("- **Status (")]
assert len(si) == 1, si
old = "Still open for Mirko: the R arm (ResNet-18; its 30 minute distillation tells whether it copies the teacher better than 0.72 / 0.67), second seeds of stage B, a B3 with gradient through the older frame (new code first), or stopping the student route."
assert lines[si[0]].count(old) == 1, "status sentence not found exactly once"
new = ("**2026-11-09 08:50: Mirko chose (D67) the seed twin of stage B and the R arm, to run now; the set-up, predictions and rules are written in \"Seed twin and R arm\" below, before any run of them.** "
       "Still open for Mirko afterwards: a B3 with gradient through the older frame (new code first), or stopping the student route.")
lines[si[0]] = lines[si[0]].replace(old, new)

out = "\n".join(lines)
if crlf:
    out = out.replace("\n", "\r\n")
open(P, "wb").write(out.encode("utf-8"))
print("bullet inserted after line", bi[0] + 1, "; status line", si[0] + 1, "; bytes", len(out.encode("utf-8")), "crlf", crlf)
