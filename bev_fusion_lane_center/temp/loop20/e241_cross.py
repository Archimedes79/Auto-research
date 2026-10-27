"""E241 pair 3 closure: the four cross-pairings (seed 2 / 3 of j1rc against seed 2 / 3 of j1eq) per checkpoint, lanes found and Q from runs/eval/<name>/val.json."""
import json
R = "runs/eval"
def m(n, k):
    d = json.load(open(f"{R}/{n}/val.json"))
    return (d.get("metrics") or d)[k]
for p in (25, 50, 75):
    for k in ("lanes_found", "Q"):
        row = []
        for a in (2, 3):
            for c in (2, 3):
                row.append(f"rc{a}-eq{c} {m(f'e24_j1rc_s{a}_at{p}', k) - m(f'e24_j1eq_s{c}_at{p}', k):+.4f}")
        print(f"at {p}% {k}: " + "  ".join(row))
for k in ("lanes_found", "Q"):
    print("final", k, "  ".join(f"rc{a}-eq{c} {m(f'e24_j1rc_s{a}', k) - m(f'e24_j1eq_s{c}', k):+.4f}" for a in (2, 3) for c in (2, 3)))
