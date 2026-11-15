"""scratch (Image Encoder, clean-base round): cut section K (student_frames 'all', stage B2) out of tests/test_student_chain.py and fix the lines of the docstring and of section E that name it.
Asserts every anchor once."""
P = "tests/test_student_chain.py"
raw = open(P, "rb").read().decode("utf-8")
crlf = "\r\n" in raw
lines = raw.replace("\r\n", "\n").split("\n")


def find(prefix, start=0):
    idx = [i for i in range(start, len(lines)) if lines[i].startswith(prefix)]
    assert len(idx) == 1, (prefix, idx)
    return idx[0]


# 1. section K: from its banner to the line before the final summary
k0 = find("# ------------------------------------------------------------------------------------------------ K")
end = find('print(f"\\ntest_student_chain:')
assert lines[k0 + 1].startswith('print("K: student_frames'), lines[k0 + 1]
del lines[k0:end]                                           # the blank line before the summary stays: lines[k0 - 1] is blank already
if lines[k0 - 1] == "" and lines[k0] != "":
    lines.insert(k0, "")

# 2. docstring: the K entry, and the E entry's mention
d0 = find("  K  student_frames 'all'")
d1 = find("Run from the bev_fusion folder:")
del lines[d0:d1]
e = find("  E  config: each refusal")
assert "student_frames 'all' accepted with a student and refused without one; " in lines[e]
lines[e] = lines[e].replace("student_frames 'all' accepted with a student and refused without one; ", "student_frames other than 'current' refused (the variant 'all' was cut with t22 stage B2); ")

# 3. section E: the two 'all' checks
a = find('check(not raises(ValueError, lambda: check_config(ModelConfig(**dict(DUAL, camera_student="stub", student_frames="all", student_trainable=True)))),')
assert 'accepted (section K)' in lines[a + 1]
lines[a:a + 2] = ['check(raises(ValueError, lambda: check_config(ModelConfig(**dict(DUAL, camera_student="stub", student_frames="all", student_trainable=True))), "student_frames", "only \'current\'"),',
                  '      "student_frames \'all\' is refused (the variant was cut with t22 stage B2: tag bf/t22.B2_s0)")']
b = find('check(raises(ValueError, lambda: check_config(ModelConfig(**dict(DUAL, student_frames="all"))), "student_frames", "without camera_student")')
del lines[b]

# 4. the first line of the docstring names the option
assert "student_init / student_trainable / student_frames," in lines[0]

out = "\n".join(lines)
if crlf:
    out = out.replace("\n", "\r\n")
open(P, "wb").write(out.encode("utf-8"))
print("section K cut: lines", k0 + 1, "to", end, "; file now", len(lines), "lines")
