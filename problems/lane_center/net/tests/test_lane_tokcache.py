"""Tests of the predicted-node cache (`net/prep/lane_tokcache.py`) and its loader (`net/prep/lane_ptok_data.py`).

    python net/tests/test_lane_tokcache.py            # the selftests of the two modules, then the hand mutants of the label, selection, comparison and loader code: every mutant must make the selftest of its module FAIL
    python net/tests/test_lane_tokcache.py selftests  # the selftests only (about 1 minute)

A mutant = one textual change of the source (a wrong assignment, a swapped link direction, the contraction removed, a padding token that becomes a node, ...); the mutated copy runs in a temporary folder with BFLC_ROOT
pointing at this project, four at a time.
"""
import concurrent.futures as cf
import os
import subprocess
import sys
import tempfile

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "net", "prep")

TC, PT = "lane_tokcache.py", "lane_ptok_data.py"
MUTANTS = [
    (TC, "tokens are taken in reverse order (not by descending existence)", "for t in range(min(K, near.shape[0])):", "for t in reversed(range(min(K, near.shape[0]))):"),
    (TC, "a token takes the FARTHEST near node", "g = int(np.argmin(np.where(cand, d[t], np.inf)))", "g = int(np.argmax(np.where(cand, d[t], -np.inf)))"),
    (TC, "a token takes the LOWEST near node", "g = int(np.argmin(np.where(cand, d[t], np.inf)))", "g = int(np.argmax(cand))"),
    (TC, "a node can be assigned twice", "cand = near[t] & ~used", "cand = near[t]"),
    (TC, "the heading cosine is ignored", "& (d <= r) & (cs >= cos), d", "& (d <= r), d"),
    (TC, "the cosine bound is exclusive", "& (d <= r) & (cs >= cos), d", "& (d <= r) & (cs > cos), d"),
    (TC, "the radius is exclusive", "& (d <= r) & (cs >= cos), d", "& (d < r) & (cs >= cos), d"),
    (TC, "the contraction is removed (every GT node counts as covered)", "sub = sub_table(table, covered)", "sub = sub_table(table, np.ones(M, bool)); covered = np.ones(M, bool)"),
    (TC, "covered by a token beyond K", "covered = near[:K].any(0)", "covered = near.any(0)"),
    (TC, "targets beyond K", "for j in np.flatnonzero(near[:K, b]):", "for j in np.flatnonzero(near[:, b]):"),
    (TC, "the link direction is swapped", "a, b = int(kept[sub[\"src\"][l]]), int(kept[sub[\"dst\"][l]])", "b, a = int(kept[sub[\"src\"][l]]), int(kept[sub[\"dst\"][l]])"),
    (TC, "a link without a source token is not skipped", "            nosrc += 1\n            continue\n", "            nosrc += 1\n"),
    (TC, "a link without a source token is not counted", "            nosrc += 1\n", "            nosrc += 0\n"),
    (TC, "a token is its own target", "            if j == i:\n                continue\n", ""),
    (TC, "a pair of two classes keeps the short one", "if plong[(i, int(j))] or not lg:", "if not plong[(i, int(j))] or lg:"),
    (TC, "a pair of two classes is not counted", "conflict += int(plong[(i, int(j))] != lg)", "conflict += 0"),
    (TC, "the type priority is reversed", "for code, cols in ((6, (4, 5, 6, 7)), (5, (1,)), (4, (0,)), (3, (3,)), (2, (2,))):", "for code, cols in ((2, (2,)), (3, (3,)), (4, (0,)), (5, (1,)), (6, (4, 5, 6, 7))):"),
    (TC, "exit beats stop", "(5, (1,)), (4, (0,)),", "(4, (0,)), (5, (1,)),"),
    (TC, "stop beats merge", "(4, (0,)), (3, (3,)),", "(3, (3,)), (4, (0,)),"),
    (TC, "uout and uin of the node are swapped", "uout[gt[got]], uin[gt[got]]], 1)", "uin[gt[got]], uout[gt[got]]], 1)"),
    (TC, "the tokens are sorted ascending", "descending=True, stable=True", "descending=False, stable=True"),
    (TC, "the active threshold is 0.5", "g1(ep) > P_ACTIVE", "g1(ep) > 0.5"),
    (TC, "the delivered threshold is exclusive", "deliv = valid_k & (ep_k >= 0.5)", "deliv = valid_k & (ep_k > 0.5)"),
    (TC, "n_e20 counts above 0.3", "n_e20=(ep >= 0.2).sum(1)", "n_e20=(ep >= 0.3).sum(1)"),
    (TC, "the probabilities are rounded to half precision", "e_p=g1(ep)", "e_p=g1(ep).half().float()"),
    (TC, "uout and uin are swapped in the tokens", "uout=g1(no[\"uout\"].float()), uin=g1(no[\"uin\"].float())", "uout=g1(no[\"uin\"].float()), uin=g1(no[\"uout\"].float())"),
    (TC, "s is taken from uout", "s=g1(no[\"s\"].float()), s_p", "s=g1(no[\"uout\"].float()), s_p"),
    (TC, "the flag logits are reversed", "flags=gc(no[\"flags\"].float()), flags_p", "flags=gc(no[\"flags\"].float().flip(1)), flags_p"),
    (TC, "off is taken from pos", "off=gc(no[\"off\"].float())", "off=gc(no[\"pos\"].float())"),
    (TC, "the fixture passes everything", "ok = not bad and all(v == 0.0 for v in worst.values())", "ok = True"),
    (TC, "the fixture does not compare the headings", '("dirv", d["dirv"][k], f["dirv"][i, j]), ', ""),
    (TC, "the fixture does not look behind the stored nodes", "if n < cache.K and nc == n and n < b_cap and (f[\"e_p\"][i, n:] >= e_min).any():", "if False:"),
    (TC, "the fixture ignores the order", "if not np.array_equal(np.asarray(d[\"e\"][:nc], np.float64), f[\"e_p\"][i, :nc].astype(np.float64)):", "if False:"),
    (TC, "a boundary tie at any existence", "all(ed[c_] == last for c_ in only_d) and all(ec[c_] == last for c_ in only_c)", "True"),
    (TC, "a boundary tie without a full active set", "if n >= b_cap and nc == n and all(", "if all("),
    (TC, "the look-behind check ignores the cap of the active set", "if n < cache.K and nc == n and n < b_cap and", "if n < cache.K and nc == n and"),
    (TC, "the gate is read from the wrong pair", "g_ = bool(allowed[i, j]) if allowed is not None else False", "g_ = bool(allowed[j, i]) if allowed is not None else False"),
    (TC, "the stats use the stored K for every budget", "np.array([near[:min(k, T)].any(0).sum() for k in ks], np.int32)", "np.array([near[:K].any(0).sum() for k in ks], np.int32)"),
    (TC, "gt_out counts the contracted links", "gt_out=np.int32(len(src) - both(covered))", "gt_out=np.int32(len(sub[\"src\"]))"),
    (TC, "the cache comparison ignores the tokens", "x, y = np.asarray(ca.f[k][:n]), np.asarray(cb.f[k][:n])", "x, y = np.asarray(ca.f[k][:n]), np.asarray(ca.f[k][:n])"),
    (PT, "a padding token becomes a node", "keep = np.asarray(keep, bool) & np.asarray(frame[\"valid\"], bool)", "keep = np.asarray(keep, bool)"),
    (PT, "a removed source token keeps its pair", "k = keep[i] & keep[j]", "k = keep[j]"),
    (PT, "a removed target token keeps its pair", "k = keep[i] & keep[j]", "k = keep[i]"),
    (PT, "the token drop does not touch valid", "valid = np.array(f[\"valid\"][i]) & keep", "valid = np.array(f[\"valid\"][i])"),
    (PT, "a dropped token keeps its type", "ytype=np.where(keep, np.array(f[\"gt_type\"][i]), 0).astype(np.int64)", "ytype=np.array(f[\"gt_type\"][i]).astype(np.int64)"),
    (PT, "the fitted scale goes to the wrong pair", "y[i, j], yl[i, j], ys[i, j] = 1, ylong[ok_idx], yscale[ok_idx]", "y[i, j], yl[i, j], ys[i, j] = 1, ylong[ok_idx[::-1]], yscale[ok_idx[::-1]]"),
    (PT, "ylong marks every pair", "yl[i, j] = (np.linalg.norm(pos[j] - pos[i], axis=-1) > QT.SHORT)", "yl[i, j] = True"),
    (PT, "ylong from the wrong token distance", "yl[i, j] = (np.linalg.norm(pos[j] - pos[i], axis=-1) > QT.SHORT)", "yl[i, j] = (np.linalg.norm(pos[j] - pos[i], axis=-1) > 2 * QT.SHORT)"),
    (PT, "a chord of exactly 8 m is long", "yl[i, j] = (np.linalg.norm(pos[j] - pos[i], axis=-1) > QT.SHORT)", "yl[i, j] = (np.linalg.norm(pos[j] - pos[i], axis=-1) >= QT.SHORT)"),
    (PT, "ylong uses the chord of the wrong pair", "yl[i, j] = (np.linalg.norm(pos[j] - pos[i], axis=-1) > QT.SHORT)", "yl[i, j] = (np.linalg.norm(pos[i] - pos[0], axis=-1) > QT.SHORT)"),
    (PT, "a GT field in the read-out input", "tokens_np=dict(pos=", "tokens_np=dict(gt_id=np.array(f[\"gt_id\"][i]), pos="),
    (PT, "the link direction is swapped in the prediction", "lk = dict(src=new[src[ok]], dst=new[dst[ok]],", "lk = dict(src=new[dst[ok]], dst=new[src[ok]],"),
]


def run_selftests():
    for mod in (TC, PT):
        r = subprocess.run([sys.executable, os.path.join(DATA, mod), "selftest"], capture_output=True, text=True, env=dict(os.environ, BFLC_ROOT=ROOT))
        assert r.returncode == 0, f"{mod} selftest failed:\n{r.stdout[-800:]}{r.stderr[-1500:]}"
        print(r.stdout.strip().splitlines()[-1], flush=True)


def one(args):
    k, (mod, name, a, b), tmp = args
    src = open(os.path.join(DATA, mod), encoding="utf-8").read()
    if src.count(a) != 1:
        return name, "NOT APPLIED"
    d = os.path.join(tmp, str(k))
    os.makedirs(d)
    p = os.path.join(d, mod)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(src.replace(a, b))
    r = subprocess.run([sys.executable, p, "selftest"], capture_output=True, text=True, env=dict(os.environ, BFLC_ROOT=ROOT))
    return name, "killed" if r.returncode != 0 else "SURVIVED"


def run_mutants(workers=4):
    left = []
    with tempfile.TemporaryDirectory() as tmp, cf.ThreadPoolExecutor(workers) as ex:
        for name, res in ex.map(one, [(k, m, tmp) for k, m in enumerate(MUTANTS)]):
            print(f"{res:<12} {name}", flush=True)
            if res != "killed":
                left.append(name)
    print(f"{len(MUTANTS) - len(left)} of {len(MUTANTS)} killed", flush=True)
    return left


if __name__ == "__main__":
    run_selftests()
    if len(sys.argv) < 2 or sys.argv[1] != "selftests":
        left = run_mutants()
        assert not left, f"mutants survived: {left}"
    print("test_lane_tokcache ok")
