"""One-off: find how a Google Drive folder page embeds its entries (names containing gts / imgs / annotations / .tar / .zip)."""
import re
import sys

p = open(sys.argv[1], encoding="utf-8", errors="replace").read()
for key in ("gts", "imgs", "annotations", ".tar", ".zip", ".json"):
    ms = list(re.finditer(re.escape(key), p))
    print(f"{key}: {len(ms)} occurrences")
    for m in ms[:2]:
        print("   ", p[max(0, m.start() - 200): m.end() + 200].replace("\n", " "))
