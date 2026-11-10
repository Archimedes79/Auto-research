"""Plausibility screen for the numbers of a document: which decimal numbers in the text do not appear in the source files? It is not a proof.

    python check_numbers.py <document.md|.html|.txt> <source1> [<source2> ...] [--show 40] [--exact] [--strict]
    python check_numbers.py <document.md> LEDGER.csv metrics.json --exact --strict      # same digits as the source; fails on a number not found, proves nothing about the others
    python check_numbers.py --selftest

Sources are the files the numbers should come from (metrics.json, digests, review files, LEDGER.csv).
A decimal number of the document (0.746, 13.5, 31 ms is ignored: only numbers with a decimal point) counts as found when a source contains the same number;
without --exact also a number that rounds to it (0.7461 supports 0.746 and 0.75). "Found" is a weak statement: it cannot tell a right number from a wrong one
that happens to exist elsewhere in the sources, and a large pool finds random numbers by chance (random 2-decimal numbers against the 900 decimals of a ledger:
95 % found with rounding, 61 % with --exact; 3 decimals: 45 % and 40 %). Use --exact where the document quotes the digits of the source; it only finds numbers
that exist nowhere. Everything not found is listed with its line, for a human or the fresh reviewer to look at.
Advisory, exit code 0; with --strict exit code 1 when a number is not found, or when the document or the sources hold no decimal number at all (then nothing
was checked, which is not a pass).
"""
import argparse
import html
import re
import sys

NUM = re.compile(r"(?<![\w.])[-+]?\d+\.\d+(?![\w.]*\d)")


def numbers(text):
    return [m.group(0).lstrip("+") for m in NUM.finditer(text)]


def scan(raw, pool, exact=False):
    """(number of decimal numbers in the text, [(line, number, context)] of those not found in the pool of source numbers)"""
    by_dec = {}                                       # round-aware lookup by decimals: built once
    for tok in pool:
        v = float(tok)
        for dec in (1, 2, 3, 4):
            by_dec.setdefault(dec, set()).add(round(v, dec))
    total, out = 0, []
    for ln, line in enumerate(raw.splitlines(), 1):
        for tok in numbers(line):
            total += 1
            dec = len(tok.split(".")[1])
            if not (tok in pool or (not exact and dec in by_dec and round(float(tok), dec) in by_dec[dec])):
                out.append((ln, tok, line.strip()[:110]))
    return total, out


def selftest():
    pool = set(numbers('{"S": 0.74612, "Q": 0.5, "d": -0.012}'))
    text = "S 0.746 and 0.99 here, Q 0.5, delta -0.012, 2.5 ms\nand 0.7461\n"
    total, bad = scan(text, pool)
    assert total == 6 and [b[1] for b in bad] == ["0.99", "2.5"], (total, bad)                # 0.746 is found by rounding
    total, bad = scan(text, pool, exact=True)
    assert [b[1] for b in bad] == ["0.746", "0.99", "2.5", "0.7461"] and bad[-1][0] == 2, bad      # --exact: the digits of the source, 0.74612 is not 0.7461
    assert scan("no decimals, 12 ms", pool) == (0, []) and scan("0.5", set()) == (1, [(1, "0.5", "0.5")])
    import os
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        def run(doc, src, *flags):
            for name, body in (("doc.md", doc), ("src.json", src)):
                with open(os.path.join(t, name), "w", encoding="utf-8") as f:
                    f.write(body)
            r = subprocess.run([sys.executable, os.path.abspath(__file__), os.path.join(t, "doc.md"), os.path.join(t, "src.json"), *flags], capture_output=True, text=True)
            return r.returncode, r.stdout
        assert run("Q 0.5", '{"Q": 0.5}', "--strict", "--exact")[0] == 0
        assert run("Q 0.6", '{"Q": 0.5}', "--strict")[0] == 1 and run("Q 0.6", '{"Q": 0.5}')[0] == 0, "--strict fails on a number not found; without it the tool is advisory"
        assert run("Q 0.50", '{"Q": 0.5}')[0] == 0 and run("Q 0.5001", '{"Q": 0.5}', "--strict", "--exact")[0] == 1
        code, out = run("only 12 ms and 3 seeds", '{"Q": 0.5}', "--strict")
        assert code == 1 and "no decimal number" in out, (code, out)                            # nothing to check is no pass
        code, out = run("Q 0.5", '{"n": 12}', "--strict")
        assert code == 1 and "no decimal number" in out, (code, out)                            # sources without a decimal: nothing could be found
        assert run("only 12 ms", '{"Q": 0.5}')[0] == 0, "advisory without --strict"
    print("check_numbers selftest passed")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("doc", nargs="?")
    ap.add_argument("sources", nargs="*")
    ap.add_argument("--show", type=int, default=40)
    ap.add_argument("--exact", action="store_true", help="a number counts as found only with the same digits as in a source (no rounding match)")
    ap.add_argument("--strict", action="store_true", help="exit 1 when a number is not found or nothing could be checked")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_intermixed_args(argv)      # flags and sources in any order, as before the rewrite (doc.md --strict src.json)
    if a.selftest:
        return selftest()
    if not (a.doc and a.sources):
        ap.error("name the document and at least one source")
    pool = set()
    for p in a.sources:
        pool.update(numbers(open(p, encoding="utf-8", errors="replace").read()))
    raw = open(a.doc, encoding="utf-8", errors="replace").read()
    if a.doc.lower().endswith((".html", ".htm")):     # markup is removed from a page only (a text "Q < 0.5 ... -> better" is no tag), line breaks stay: lines match the file
        blank = lambda m: " " + "\n" * m.group(0).count("\n")  # noqa: E731
        raw = re.sub(r"<(script|style)[^>]*>.*?</\1>", blank, raw, flags=re.S)
        raw = html.unescape(re.sub(r"<[^>]+>", blank, raw))
    total, out = scan(raw, pool, a.exact)
    bad = len(out)
    print(f"{a.doc}: {total} decimal numbers, {total - bad} found in the sources, {bad} not found")
    for ln, tok, ctx in out[: a.show]:
        print(f"  line {ln}: {tok}   | {ctx}")
    if bad > a.show:
        print(f"  ... {bad - a.show} more")
    empty = not total or not pool
    if empty:
        print(f"WARNING: {'the document holds' if not total else 'the sources hold'} no decimal number: nothing was checked")
    return 1 if a.strict and (bad or empty) else 0


if __name__ == "__main__":
    sys.exit(main())
