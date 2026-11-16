"""Soften the note of 03_numbers.json (scratch, run once)."""
import json, os
P = os.path.join(os.environ.get("BFLC_ROOT", os.getcwd()), "results", "Ergebnis_Spurmitte_2026-11-15", "parts", "03_numbers.json")
d = json.load(open(P, encoding="utf-8"))
d["note"] = ("register of the headline numbers of the tables of 03_results.md with their source (a result json, '#' = key path, or a card with its section; "
             "'derived' = computed from registered numbers); table N registers the headline numbers of the night; every other number of tables A and C cites its "
             "card section in its row (not every number is registered here); VAL = 873 frames / 22 scenes, TEST = 833 frames / 21 scenes")
json.dump(d, open(P, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("ok")
