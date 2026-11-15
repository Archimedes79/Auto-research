"""scratch: replace section J of tests/test_student_chain.py by temp/t22/newJ.txt (the text between the J header and the final print)"""
p = "tests/test_student_chain.py"
s = open(p, encoding="utf-8").read()
a = s.index("# ------------------------------------------------------------------------------------------------ J")
b = s.index("print(f\"\\ntest_student_chain:")
new = open("temp/t22/newJ.txt", encoding="utf-8").read()
open(p, "w", encoding="utf-8").write(s[:a] + new + s[b:])
print("replaced", a, b)
