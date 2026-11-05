"""First-pass numbers check for a document: which numbers in the text do not appear in the source files?

    python check_numbers.py <document.md|.html|.txt> <source1> [<source2> ...] [--show 40] [--strict]

Sources are the files the numbers should come from (metrics.json, digests, review files, LEDGER.csv).
A decimal number of the document (0.746, 13.5, 31 ms is ignored: only numbers with a decimal point) counts as supported when a source contains the same
number, or a number that rounds to it (0.7461 supports 0.746 and 0.75).  Everything else is listed with its line, for a human or the fresh reviewer to look at.
It cannot tell a right number from a wrong one that happens to exist elsewhere in the sources; it only finds numbers that exist nowhere. Advisory, exit code 0;
with --strict exit code 1 when a number is not found (a gate for tables built from metric cards).
"""
import argparse
import html
import re
import sys

NUM = re.compile(r"(?<![\w.])[-+]?\d+\.\d+(?![\w.]*\d)")


def numbers(text):
    return [m.group(0).lstrip("+") for m in NUM.finditer(text)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("doc")
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--show", type=int, default=40)
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()
    pool = set()
    for p in a.sources:
        pool.update(numbers(open(p, encoding="utf-8", errors="replace").read()))
    raw = open(a.doc, encoding="utf-8", errors="replace").read()
    if a.doc.lower().endswith((".html", ".htm")):     # markup is removed from a page only (a text "Q < 0.5 ... -> better" is no tag), line breaks stay: lines match the file
        blank = lambda m: " " + "\n" * m.group(0).count("\n")  # noqa: E731
        raw = re.sub(r"<(script|style)[^>]*>.*?</\1>", blank, raw, flags=re.S)
        raw = html.unescape(re.sub(r"<[^>]+>", blank, raw))
    # round-aware lookup by decimals: build rounded sets once
    by_dec = {}
    for tok in pool:
        v = float(tok)
        for dec in (1, 2, 3, 4):
            by_dec.setdefault(dec, set()).add(round(v, dec))
    total = bad = 0
    out = []
    for ln, line in enumerate(raw.splitlines(), 1):
        for tok in numbers(line):
            total += 1
            dec = len(tok.split(".")[1])
            ok = tok in pool or (dec in by_dec and round(float(tok), dec) in by_dec[dec])
            if not ok:
                bad += 1
                out.append((ln, tok, line.strip()[:110]))
    print(f"{a.doc}: {total} decimal numbers, {total - bad} found in the sources, {bad} not found")
    for ln, tok, ctx in out[: a.show]:
        print(f"  line {ln}: {tok}   | {ctx}")
    if bad > a.show:
        print(f"  ... {bad - a.show} more")
    return 1 if a.strict and bad else 0


if __name__ == "__main__":
    sys.exit(main())
