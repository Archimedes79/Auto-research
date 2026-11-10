"""scratch (Image Encoder, clean-base round): repair the working tests/README.md row of test_student_chain.py after the critic found my regex replacement had inserted a literal backslash-1,
and correct the tag that holds the mutation scripts (bf/pitch-scratch). Plain str.replace, each anchor exactly once."""
P = "bev_fusion/tests/README.md"
raw = open(P, "rb").read().decode("utf-8")
crlf = "\r\n" in raw
t = raw.replace("\r\n", "\n")
fixes = [
    ("| \\1, the cached model's frame-0 splice",
     "| `test_student_chain.py` | the camera student in the chain (t22 stage B, lane branch included: `ModelConfig.camera_student` / `student_init` / `student_trainable`, the cached model's frame-0 splice"),
    ("are in the run tags, e.g. `bf/t22.B2_s0`)", "are in the tag `bf/pitch-scratch`)"),
]
for old, new in fixes:
    assert t.count(old) == 1, (old[:50], t.count(old))
    t = t.replace(old, new)
open(P, "wb").write((t.replace("\n", "\r\n") if crlf else t).encode("utf-8"))
print("fixed", len(fixes), "anchors")
