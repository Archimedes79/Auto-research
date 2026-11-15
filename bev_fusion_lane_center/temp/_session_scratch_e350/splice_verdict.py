p = "evaluation/jwire_verdict.py"
s = open(p, encoding="utf-8").read()
i, j = s.index("def _selftest():"), s.index("def main(")
new = open("temp/_session_scratch_e350/selftest_verdict_new.txt", encoding="utf-8").read()
open(p, "w", encoding="utf-8").write(s[:i] + new + s[j:])
