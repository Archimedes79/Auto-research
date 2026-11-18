"""scratch: replace (or insert before `selftest`) the top-level functions of NEW in TARGET, by name; usage: splice.py TARGET NEW"""
import re
import sys

target, new = sys.argv[1], sys.argv[2]
src = open(target, encoding="utf-8").read().split("\n")
blocks, cur = {}, None
for line in open(new, encoding="utf-8").read().split("\n"):
    m = re.match(r"def (\w+)\(", line)
    if m:
        cur = m.group(1)
        blocks[cur] = []
    if cur is not None:
        blocks[cur].append(line)
for name, body in blocks.items():
    while body and body[-1] == "":
        body.pop()
    start = next((i for i, l in enumerate(src) if re.match(rf"def {name}\(", l)), None)
    if start is None:                                                       # a new function: before `def selftest`
        at = next(i for i, l in enumerate(src) if l.startswith("def selftest("))
        src[at:at] = body + ["", ""]
        print("inserted", name)
        continue
    end = next(i for i in range(start + 1, len(src)) if re.match(r"(def |class |if __name__|# -----)", src[i]))
    src[start:end] = body + ["", ""]
    print("replaced", name, start + 1, end)
open(target, "w", encoding="utf-8").write("\n".join(src))
