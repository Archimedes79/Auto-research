"""throw-away: the critic's surviving mutants of the live read-out, each applied in the module's own globals (`mutated`) and run against test_live of the scratch test copy; a mutant is KILLED when test_live fails with an
AssertionError or the test's "accepted although it must be refused" RuntimeError"""
import importlib.util
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("jtoken_eval", os.path.join(HERE, "e372r_jtoken_eval.py"))
E = importlib.util.module_from_spec(spec)
sys.modules["jtoken_eval"] = E
spec.loader.exec_module(E)
spec = importlib.util.spec_from_file_location("e372r_test", os.path.join(HERE, "e372r_test_jtoken_eval.py"))
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)

MUTANTS = [
    ("B' on the whole head batch (no --bs chunks)", E.live_dump, "tks = [src.tokens(b[\"x\"][c:c + bs])[1] for c in range(0, len(b[\"x\"]), bs)]", "tks = [src.tokens(b[\"x\"])[1]]"),
    ("tokens_exact against the frames of the batch's own length (off by batch)", E.live_dump, "idx[n * HEAD_BS:(n + 1) * HEAD_BS]", "idx[n * len(b[\"x\"]):(n + 1) * len(b[\"x\"])]"),
    ("the frame-order guard removed", E.live_dump, "assert list(data.tokens) == list(ds.tokens)", "assert True or list(data.tokens) == list(ds.tokens)"),
    ("anatomy --live reads the cache's labels", E.anatomy, "gid, uout, uin = ([x[k] for x in labs] for k in (\"gt_id\", \"uout\", \"uin\"))", "gid, uout, uin = (np.asarray(ds.c.f[k][idx]) for k in (\"gt_id\", \"uout\", \"uin\"))"),
    ("anatomy --live: the cache's y and ytype", E.anatomy, "item = lambda k, i: dict(tokens_np=r[\"toks\"][k], ytype=labs[k][\"ytype\"].astype(np.int64), y=pair_matrix(labs[k][\"ypair\"], ds.K))", "item = lambda k, i: dict(ds[int(i)], tokens_np=r[\"toks\"][k])"),
    ("anatomy --live: a CPU forward on the cache for the concentration", E.anatomy, "pool_concentration([f[\"conc\"] for f in r[\"fr\"]]) if r[\"live\"] else", "pool_concentration([f[\"conc\"] for f in r[\"fr\"]]) if False else"),
    ("two live dumps of another B' batch paired", E.same_tokens, "    elif r.get(\"live\") and rr.get(\"live\"):", "    elif False:"),
    ("check_b without the b_sha256 check", E.check_b, "assert \"b_sha256\" not in ck[\"args\"] or", "assert True or"),
    ("check_b without the frozen-part check", E.check_b, "assert got[\"sha_frozen\"] == base[\"sha_frozen\"]", "assert True or got[\"sha_frozen\"] == base[\"sha_frozen\"]"),
    ("read_dump accepts a joint run on the cache path", E.read_dump, "assert live or not joint_run(ck)", "assert True or not joint_run(ck)"),
    ("dump accepts a joint run on the cache path", E.dump, "assert a.live or not joint_run(ck)", "assert True or not joint_run(ck)"),
    ("fixture accepts a joint run", E.fixture, "assert not joint_run(torch.load(", "assert True or not joint_run(torch.load("),
    ("a z run with a b_state refused (train_nodes read from the b_state)", E.check_b, "    if ck[\"args\"].get(\"train_nodes\"):", "    if ck[\"args\"].get(\"train_nodes\") or \"b_state\" in ck:"),
]
want = [m for m in MUTANTS if not sys.argv[1:] or str(MUTANTS.index(m)) in sys.argv[1:]]
left = []
for name, fn, old, new in want:
    try:
        with T.mutated(fn, old, new):
            T.test_live()
        res = "SURVIVED"
    except (AssertionError, RuntimeError) as e:
        res = "killed" if isinstance(e, AssertionError) or "accepted although" in str(e) else "CRASHED " + str(e)[:120]
    except Exception as e:  # noqa: BLE001
        res = f"CRASHED {type(e).__name__}: {str(e)[:120]} | " + traceback.format_exc().strip().splitlines()[-3][:120]
    print(f"{res:<10} {name}", flush=True)
    if res != "killed":
        left.append(name)
print(f"{len(want) - len(left)} of {len(want)} killed; left: {left}")
