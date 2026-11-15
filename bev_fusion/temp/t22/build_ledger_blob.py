"""scratch: the ledger files as they should be committed by the t22 chat: the committed (HEAD) files, my rows replaced IN PLACE by the working versions, new rows of mine appended
(other chats' uncommitted rows stay out; HEAD order is kept so the diff shows only my rows). Reads temp/t22/head_LEDGER.{csv,md} (dumped from HEAD) and the working files;
writes temp/t22/blob_LEDGER.{csv,md}."""
import csv, io, re
MINE = {"t22.A1_s0", "t22.A2_s0", "t22.standin_pair4", "t22.twin_pair4", "smoke.t22_B_s0", "smoke.t22_B_s0b", "smoke.t22_B_s0c", "smoke.t21_pair4", "t22.B_s0", "t22.B_s0.eval",
        "smoke.t22_B2_s0", "t22.B2_s0", "t22.B2_s0.eval", "diag.t20_L4on", "diag.t20_L4off"}
D = "temp/t22/"
R = "records/"
# csv
head = list(csv.DictReader(open(D + "head_LEDGER.csv", newline="", encoding="utf-8")))
cols = list(csv.DictReader(open(D + "head_LEDGER.csv", newline="", encoding="utf-8")).fieldnames)
work = list(csv.DictReader(open(R + "LEDGER.csv", newline="", encoding="utf-8")))
assert list(csv.DictReader(open(R + "LEDGER.csv", newline="", encoding="utf-8")).fieldnames) == cols, "header differs"
wmap = {r["id"]: r for r in work if r["id"] in MINE}
head_ids = {r["id"] for r in head}
out = [wmap.get(r["id"], r) if r["id"] in MINE else r for r in head]
new = [r for r in work if r["id"] in MINE and r["id"] not in head_ids]
out += new
buf = io.StringIO()
w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
w.writeheader()
w.writerows(out)
open(D + "blob_LEDGER.csv", "w", newline="", encoding="utf-8").write(buf.getvalue())
changed = [r["id"] for r in head if r["id"] in wmap and wmap[r["id"]] != r]
print("csv: HEAD rows", len(head), "-> blob rows", len(out), "; new rows:", [r["id"] for r in new], "; rows of mine in HEAD that change:", changed)
print("my ids without a working row:", sorted(MINE - set(wmap)))
# md
hm = open(D + "head_LEDGER.md", encoding="utf-8", newline="").read().split("\n")
wm = open(R + "LEDGER.md", encoding="utf-8", newline="").read().split("\n")
pat = re.compile(r"^\| (%s) \|" % "|".join(re.escape(i) for i in sorted(MINE, key=len, reverse=True)))
wmd = {}
for ln in wm:
    m = pat.match(ln)
    if m:
        wmd[m.group(1)] = ln
out_md, seen = [], set()
for ln in hm:
    m = pat.match(ln)
    if m:
        seen.add(m.group(1))
        out_md.append(wmd.get(m.group(1), ln))
    else:
        out_md.append(ln)
new_md = [wmd[i] for i in wmd if i not in seen]
idx = max(i for i, ln in enumerate(out_md) if ln.startswith("| ") and not ln.startswith("| loop") and not ln.startswith("|---"))
out_md = out_md[: idx + 1] + new_md + out_md[idx + 1:]
open(D + "blob_LEDGER.md", "w", encoding="utf-8", newline="").write("\n".join(out_md))
print("md: HEAD lines", len(hm), "-> blob lines", len(out_md), "; new md rows:", len(new_md), "; replaced in place:", len(seen))
