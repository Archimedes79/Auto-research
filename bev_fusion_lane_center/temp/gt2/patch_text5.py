p = "temp/jr/report_text.py"
s = open(p, encoding="utf-8").read()


def sub(old, new):
    global s
    assert old in s, old[:70]
    s = s.replace(old, new, 1)


sub("The plain graph realises 50.4 % (junction precision 0.95), the connectors at p >= 0.5 53.8 %, at 0.2 56.2 %, at 0.1 57.3 %, the oracle pair classifier 59.1 %, the oracle with handle scales fitted to the GT 61.5 %. The pair decision can add about five points at most; the rest is lost elsewhere.",
    "The plain graph realises 52.5 % (junction precision 0.95), the connectors at p >= 0.5 56.6 %, at 0.2 59.8 %, at 0.1 61.6 %, the oracle pair classifier 64.3 %, the oracle with handle scales fitted to the GT 67.1 %. At p >= 0.1 the pair decision has under three points left; the rest is lost elsewhere (the measure was corrected after an independent review: a first version limited the direction test to the six nearest delivered points and read about three points too low).")
sub("Where each maneuver type stands (maneuvers with both ends found, perfect pairs): straight 75 %, left 55 %, right 52 %; chord up to 10 m 81 %, 20 - 35 m 59 %, 35 - 60 m 53 %. Right turns and long connectors are the weak classes.",
    "Where each maneuver type stands (maneuvers with both ends found, perfect pairs): straight 80 %, left 64 % (78 % with fitted scales), right 60 % (70 %); chord up to 10 m 81 %, 20 - 35 m 68 %, 35 - 60 m 63 %. Turns and long connectors are the weak classes, and the ones where fitted handle scales help most.")
open(p, "w", encoding="utf-8").write(s)
print("ok")
