import os
P = os.path.join(os.environ["BFLC_ROOT"], "evaluation", "tests", "test_jtest_eval.py")
s = open(P, encoding="utf-8").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:90])
    s = s.replace(old, new)


rep('''        def dump_full(preds, rows_, path):
            order.append(("dump_full", os.path.exists(b_out[:-4] + "_head.json")))
            pickle.dump([dict(row=int(r)) for r in rows_], open(path, "wb"))
        orig_save = J.save_pickle

        def save_pickle(path, obj, overwrite=True):
            order.append(("save_pickle", os.path.exists(e_out[:-4] + "_head.json"), overwrite))
            orig_save(path, obj, overwrite)
        with patched((J, "provenance", PROV), (J, "frames_of", lambda split: frames), (J, "EXTRA_WRITABLE", (d,)), (J, "save_pickle", save_pickle), (LD, "loader", lambda *a, **k: (ds, batches)), (QP, "load_model", lambda run_dir, device: "model"),''',
    '''        def dump_full(preds, rows_, path):
            pickle.dump([dict(row=int(r)) for r in rows_], open(path, "wb"))
        orig_atomic = J.atomic

        def atomic_spy(path, overwrite=True):                                                          # every write of a dump pickle: is the sidecar there already, may it overwrite?
            if path.endswith(".pkl"):
                order.append((os.path.basename(path), overwrite, os.path.exists(path[:-4] + "_head.json")))
            return orig_atomic(path, overwrite)
        with patched((J, "provenance", PROV), (J, "frames_of", lambda split: frames), (J, "EXTRA_WRITABLE", (d,)), (J, "atomic", atomic_spy), (LD, "loader", lambda *a, **k: (ds, batches)), (QP, "load_model", lambda run_dir, device: "model"),''')
rep('''        assert order == [("dump_full", True), ("save_pickle", True, False)], "the sidecar is there before the pickle; the pair dump's pickle is written without overwriting"''',
    '''        assert order == [("b.pkl", False, True), ("e.pkl", False, True)], "each pickle is written after its sidecar and without overwriting: a crash leaves a sidecar without a pickle, never a pickle without provenance"''')

# smoke card and a card without a rehearsal check are two different refusals
rep('''rehearsal_card(files, frames=5, rehearsal_check=None), rehearsal_card(files, split="test"),''', '''rehearsal_card(files, frames=5), rehearsal_card(files, rehearsal_check=None), rehearsal_card(files, split="test"),''')
rep('''for name, bad in zip(("code drift", "failed rehearsal", "smoke card", "TEST card", "tampered digest",''', '''for name, bad in zip(("code drift", "failed rehearsal", "smoke card (frames)", "no rehearsal check", "TEST card", "tampered digest",''')
open(P, "w", encoding="utf-8").write(s)
print("ok")
