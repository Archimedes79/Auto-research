import io
p = "evaluation/jwire_check.py"
s = io.open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:90], s.count(a))
    s = s.replace(a, b)
rep('''            with mock.patch.object(sys.modules[__name__], "live_check", lambda *x, **k: calls.append(x + (k.get("bs"),)) or (True, 24, {})), contextlib.redirect_stdout(io.StringIO()):
                assert main(["live", "--seed", "1", "--pair-run", "e324_s_s1", "--ends", "att", "--bs", "8"]) == 0
            assert calls[-1][0] == 1 and calls[-1][-1] == 8, calls[-1]                                         # --bs 8 reaches live_check (E358: the dump's batch size)
            assert [c[-1] for c in calls] == [4, 4, 4, 8]                                                       # --bs defaults to 4 (E325 / E350), E358 passes --bs 8 for the dump's batch size''',
'''            with mock.patch.object(sys.modules[__name__], "live_check", lambda *x, **k: calls.append(x + (k.get("bs"), k.get("faithful"))) or (True, 24, {})), contextlib.redirect_stdout(io.StringIO()):
                assert main(["live", "--seed", "1", "--pair-run", "e324_s_s1", "--ends", "att", "--bs", "8"]) == 0
                assert main(["live", "--seed", "1", "--pair-run", "e324_s_s1", "--ends", "att", "--faithful"]) == 0
            assert calls[-2][0] == 1 and calls[-2][-2] == 8, calls[-2]                                         # --bs 8 reaches live_check (E358 attempt 3)
            assert [c[-2] for c in calls] == [4, 4, 4, 8, 4]                                                    # --bs defaults to 4 (E325 / E350)
            assert [c[-1] for c in calls] == [False, False, False, False, True], calls                          # --faithful (E358 attempt 4) reaches live_check, and only when given''')
rep('v.add_argument("--bs", type=int, default=4,', 'v.add_argument("--faithful", action="store_true", help="E358 attempt 4: the live pass replays HOW the stored dumps were made (B on a loader of its dump\'s settings, the pair model on a loader of the pair dump\'s settings, joined by row; `jwire_data.DUMP_B` / `DUMP_PAIR`; at most 64 frames); constants, not options")\n    v.add_argument("--bs", type=int, default=4,')
rep("ok, n, diff = live_check(a.seed, a.pair_run, a.max, a.workers, a.device, ends, bs=a.bs)", "ok, n, diff = live_check(a.seed, a.pair_run, a.max, a.workers, a.device, ends, bs=a.bs, faithful=a.faithful)")
rep("ends, batch size {a.bs}) on {n}", "ends, " + '{\\"faithful two-loader replay (B bs 4 exact, pair model bs 8 error)\\" if a.faithful else f\\"batch size {a.bs}\\"}' + ") on {n}") if False else None
io.open(p, "w", encoding="utf-8").write(s)
print("ok")
