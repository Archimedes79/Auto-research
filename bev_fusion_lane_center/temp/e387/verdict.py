"""E387 clause table (scratch, 2026-11-18): reads the score cards of f_s0 / f_s1 (runs/e387_f_s<S>/token_card_*.json: each card names its decode cell, its reference run and the reference cell inside), the training logs and the timing json, and prints every clause of the card's rule with its value. The verdict itself is written by hand into the card (section 3); this tool never decides.
usage: BFLC_ROOT=. python verdict.py"""
import glob
import json
import os
import sys

R = os.environ.get("BFLC_ROOT", ".")
CELLS = {}
for ln in open(os.path.join(R, "temp/e387/cells.log"), encoding="utf-8"):
    k, v = ln.split()
    CELLS[k] = v


def cell_of_card(d):
    v = d["info"]["decode_variant"]
    return f"{v['link_thr']},{v.get('link_thr_long', v['link_thr'])},{v['keep_thr']},{'on' if v['none_gate'] else 'off'}"


def ref_cell(d):
    c = d["info"]["ref_run"]["cell"]
    if isinstance(c, str):                                   # "the decode of this card": the reference is scored at the card's own cell
        return cell_of_card(d)
    return f"{c['link_thr']},{c.get('link_thr_long', c['link_thr'])},{c['keep_thr']},{'on' if c['none_gate'] else 'off'}"


def norm(c):
    a = [x for x in c.split(",")]
    return ",".join(f"{float(x):g}" if x not in ("on", "off") else x for x in a)


def cards(s):
    out = []
    for f in sorted(glob.glob(os.path.join(R, "runs", f"e387_f_s{s}", "token_card_*.json"))):
        d = json.load(open(f))
        if "ref_run" not in d["info"] or not isinstance(d["info"]["ref_run"], dict):
            continue
        out.append((norm(cell_of_card(d)), d["info"]["ref_run"]["run"], norm(ref_cell(d)), d, f))
    return out


def pick(s, own, ref, rcell):
    for c, r, rc, d, f in cards(s):
        if c == norm(own) and r == ref and rc == norm(rcell):
            return d
    return None


def blk(d, key="paired_vs_ref_run"):
    return d["info"][key] if d else None


def line(name, v):
    if v is None:
        return f"{name}: MISSING"
    return f"{name}: {v['diff']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]"


rows = {}
for s in (0, 1):
    own = next((ln.split()[1] for ln in open(os.path.join(R, "temp/e387/cells.log")) if ln.startswith(f"f_s{s} ")), None)
    print(f"== seed {s}: f's own pinned cell {own}; z's own cell {CELLS.get(f'z_s{s}')}; r's {CELLS.get(f'r_s{s}')}; common {CELLS.get('common')}")
    if own is None:
        print("   the own cell of f is not logged yet")
        continue
    sets = {"own_z": pick(s, own, f"e368_z_s{s}", CELLS[f"z_s{s}"]), "own_r": pick(s, own, f"e368_r_s{s}", CELLS[f"r_s{s}"]),
            "common_z": pick(s, CELLS["common"], f"e368_z_s{s}", CELLS["common"]), "d115_z": pick(s, "0.3,0.2,0.6,off", f"e368_z_s{s}", "0.3,0.2,0.6,off"),
            "off": pick(s, own, f"e387_f_s{s}_off", own)}
    rows[s] = sets
    for k, d in sets.items():
        if d is None:
            print(f"   {k}: card MISSING")
            continue
        p = blk(d)
        print(f"   {k}: f topo_F1 {d['metrics']['topo_F1']:.4f}  " + "  ".join(line(m, p.get(m)) for m in ("topo_F1", "junction_F1", "F1@1.0", "lanes_found", "dup_frac", "Q")))

print("\n== the clauses (card section 0)")
ok = {}
for s in (0, 1):
    if s not in rows or rows[s]["own_z"] is None:
        continue
    p = blk(rows[s]["own_z"])
    c1 = p["topo_F1"]["diff"] >= 0.005 and p["topo_F1"]["lo"] > 0
    g = (p["junction_F1"]["diff"] >= -0.010, p["F1@1.0"]["diff"] >= -0.005, p["lanes_found"]["diff"] >= -0.010, p["dup_frac"]["diff"] <= 0.010)
    cz = rows[s]["common_z"]
    c2 = cz is not None and blk(cz)["topo_F1"]["diff"] >= 0.003
    off = rows[s]["off"]
    c4 = off is not None and blk(off)["topo_F1"]["diff"] >= 0.003 and blk(off)["topo_F1"]["lo"] > 0
    cut = p["topo_F1"]["diff"] <= -0.005
    ok[s] = dict(c1=c1, guards=g, c2=c2, c4=c4, cut=cut, diff=p["topo_F1"]["diff"])
    print(f"seed {s}: (1) f - z topo F1 at own cells {p['topo_F1']['diff']:+.4f} [{p['topo_F1']['lo']:+.4f}, {p['topo_F1']['hi']:+.4f}] -> {'ok' if c1 else 'NO'}; "
          f"(2) common cell {'%+.4f' % blk(cz)['topo_F1']['diff'] if cz else 'MISSING'} -> {'ok' if c2 else 'NO'}; (3) guards junction/F1@1/lanes/dup {g}; "
          f"(4) stream - off {'%+.4f [%+.4f, %+.4f]' % tuple(blk(off)['topo_F1'][k] for k in ('diff', 'lo', 'hi')) if off else 'MISSING'} -> {'ok' if c4 else 'NO'}; CUT-side {cut}")
    r_ = rows[s]["own_r"]
    if r_ is not None:
        print(f"         f - r topo F1 {blk(r_)['topo_F1']['diff']:+.4f} [{blk(r_)['topo_F1']['lo']:+.4f}, {blk(r_)['topo_F1']['hi']:+.4f}]")
print("\n(1) both seeds:", all(v["c1"] for v in ok.values()) if len(ok) == 2 else "incomplete", "| (2):", all(v["c2"] for v in ok.values()) if len(ok) == 2 else "incomplete", "| (4):", all(v["c4"] for v in ok.values()) if len(ok) == 2 else "incomplete",
      "| guards:", all(all(v["guards"]) for v in ok.values()) if len(ok) == 2 else "incomplete", "| CUT both seeds:", all(v["cut"] for v in ok.values()) if len(ok) == 2 else "incomplete")
print("mean f - r over the seeds:", (sum(blk(rows[s]['own_r'])['topo_F1']['diff'] for s in rows if rows[s]['own_r']) / max(1, sum(1 for s in rows if rows[s]['own_r']))) if rows else "n/a")
print("(5) kill: see the training logs (u over the last steps, non-finite, loss ratio); (6) cost: see the timing json; the existence AP: temp/e387/ap.out")
