"""compare the numbers of two score cards (a re-score against its backup): exit 0 when equal"""
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
ok = a.get("numbers") == b.get("numbers") and a.get("difference") == b.get("difference")
print("EQUAL" if ok else "MISMATCH", sys.argv[1])
sys.exit(0 if ok else 1)
