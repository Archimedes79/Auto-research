"""scratch: prints the ledger row of one id (columns and values) and the head of the ledger columns"""
import csv, sys
rows = list(csv.reader(open("records/LEDGER.csv", encoding="utf-8", newline="")))
head = rows[0]
print(head)
for r in rows[1:]:
    if r and r[0] == sys.argv[1]:
        for h, v in zip(head, r):
            print(f"[{h}] {v}")
