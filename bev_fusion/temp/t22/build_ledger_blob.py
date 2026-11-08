"""scratch: the ledger files as they should be committed by the t22 chat: the committed (HEAD) files plus ONLY the rows of this chat's runs
(other chats' uncommitted rows stay out). Reads temp/t22/head_LEDGER.{csv,md} (dumped from HEAD) and the working files; writes temp/t22/blob_LEDGER.{csv,md}."""
import csv, io, re
MINE = {"t22.A1_s0", "t22.A2_s0", "t22.standin_pair4", "t22.twin_pair4", "smoke.t22_B_s0", "smoke.t22_B_s0b", "smoke.t22_B_s0c", "smoke.t21_pair4", "t22.B_s0", "t22.B_s0.eval"}
D = "temp/t22/"
R = "records/"
# csv
head = list(csv.DictReader(open(D + "head_LEDGER.csv", newline="", encoding="utf-8")))
cols = list(csv.DictReader(open(D + "head_LEDGER.csv", newline="", encoding="utf-8")).fieldnames)
work = list(csv.DictReader(open(R + "LEDGER.csv", newline="", encoding="utf-8")))
assert list(csv.DictReader(open(R + "LEDGER.csv", newline="", encoding="utf-8")).fieldnames) == cols, "header differs"
head_ids = {r["id"] for r in head}
mine_work = [r for r in work if r["id"] in MINE]
out = [r for r in head if r["id"] not in MINE] + mine_work
buf = io.StringIO()
w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
w.writeheader()
w.writerows(out)
open(D + "blob_LEDGER.csv", "w", newline="", encoding="utf-8").write(buf.getvalue())
print("csv: HEAD rows", len(head), "-> blob rows", len(out), "; my rows from the working file:", [r["id"] for r in mine_work], "; of them already in HEAD:", sorted(head_ids & MINE))
missing = MINE - {r["id"] for r in mine_work}
print("my ids without a working row:", sorted(missing))
# md
hm = open(D + "head_LEDGER.md", encoding="utf-8", newline="").read().split("\n")
wm = open(R + "LEDGER.md", encoding="utf-8", newline="").read().split("\n")
pat = re.compile(r"^\| (%s) \|" % "|".join(re.escape(i) for i in sorted(MINE, key=len, reverse=True)))
mine_md = [ln for ln in wm if pat.match(ln)]
head_md_ids = [ln for ln in hm if pat.match(ln)]
idx = max(i for i, ln in enumerate(hm) if ln.startswith("| ") and not ln.startswith("| loop") and not ln.startswith("|---"))
out_md = [ln for ln in hm if not pat.match(ln)]
idx = max(i for i, ln in enumerate(out_md) if ln.startswith("| ") and not ln.startswith("| loop") and not ln.startswith("|---"))
out_md = out_md[: idx + 1] + mine_md + out_md[idx + 1:]
open(D + "blob_LEDGER.md", "w", encoding="utf-8", newline="").write("\n".join(out_md))
print("md: HEAD lines", len(hm), "-> blob lines", len(out_md), "; my md rows:", len(mine_md), "; in HEAD already:", len(head_md_ids))
