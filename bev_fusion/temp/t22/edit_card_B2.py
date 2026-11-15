"""scratch: put the stage-B2 result into the card of t22: replaces the status bullet (line starting '- **Status') and inserts the result bullet after the Gate C0 result bullet.
Asserts every anchor exists exactly once; CRLF/LF of the file is kept."""
import io
P = "experiments/t22_student_encoder/experiment.md"
raw = open(P, "rb").read().decode("utf-8")
crlf = "\r\n" in raw
text = raw.replace("\r\n", "\n")
lines = text.split("\n")
status_new = open("temp/t22/b2_card_status.txt", encoding="utf-8").read().rstrip("\n")
bullet_new = open("temp/t22/b2_card_bullet.txt", encoding="utf-8").read().rstrip("\n")

si = [i for i, l in enumerate(lines) if l.startswith("- **Status (")]
assert len(si) == 1, si
lines[si[0]] = status_new

ci = [i for i, l in enumerate(lines) if l.startswith("  - **Gate C0 result (")]
assert len(ci) == 1, ci
assert not any(l.startswith("  - **Stage B2 result (") for l in lines), "already inserted"
lines.insert(ci[0] + 1, bullet_new)

out = "\n".join(lines)
if crlf:
    out = out.replace("\n", "\r\n")
open(P, "wb").write(out.encode("utf-8"))
print("status line", si[0] + 1, "; result bullet inserted after line", ci[0] + 1, "; file bytes", len(out.encode("utf-8")), "crlf", crlf)
