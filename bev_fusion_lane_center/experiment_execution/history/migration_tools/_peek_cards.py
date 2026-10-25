import csv, re, collections
rows = list(csv.DictReader(open("/mnt/c/ML-Experiment/report_final/text/lit16/cards.csv", encoding="utf-8", newline="")))
print(len(rows), rows[0].keys())
print(collections.Counter(r["cluster"] for r in rows))
print(collections.Counter(re.match(r"[A-Za-z]+", r["id"]).group(0) for r in rows))
for r in rows[:6] + rows[150:153]:
    print(r["id"], "|", r["code"][:260].replace("\n"," "))
n = sum(1 for r in rows if "github.com" in r["code"] or "github.com" in r["papers"])
print("with github:", n)
urls = collections.Counter()
for r in rows:
    for u in re.findall(r"https?://github\.com/[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+", r["code"] + " " + r["papers"]):
        urls[u.rstrip(".")] += 1
print(len(urls), urls.most_common(12))
print(sum(1 for r in rows if not re.search(r"arxiv|openaccess|openreview|doi|ieee|proceedings|github", r["papers"], re.I)), "without link")
