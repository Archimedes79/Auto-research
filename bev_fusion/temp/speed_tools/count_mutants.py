src = open("temp/speed_tools/mutate_host.py", encoding="utf-8").read().split("if os.environ.get(\"ONLY\")")[0]
ns = {}
exec(src, ns)
from collections import Counter
c = Counter(m[4] for m in ns["MUTANTS"] if m[1] is not None)
print(dict(c), "total", sum(c.values()))
