# applies the review fixes to tools/jarm.py by replacing the file body from the module docstring on (written as a script because the tool file changed on disk)
import re
p = "tools/jarm.py"
t = open(p, encoding="utf-8").read()
print(len(t))
