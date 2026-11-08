p = "evaluation/bev_errors.py"
s = open(p, encoding="utf-8").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)


rep('''    # the counts equal the metric's matching on random worlds''', '''    # a ghost pedestrian next to a car that WAS found is a ghost, not a class confusion; next to a car nobody found it is the confusion; a duplicate wins over the confusion
    car = box("vehicle", 20, 0)
    r3 = classify_frame([box("vehicle", 20.1, 0, 0.9), box("pedestrian", 20.8, 0.3, 0.8)], [car], ("vehicle", "pedestrian"))
    assert kinds(r3["fp"]) == ["isolated"] and not r3["fn"], r3
    r4 = classify_frame([box("pedestrian", 20.8, 0.3, 0.8)], [car], ("vehicle", "pedestrian"))
    assert kinds(r4["fp"]) == ["wrong_class"] and kinds(r4["fn"]) == ["wrong_class"], r4
    r5 = classify_frame([box("vehicle", 20.1, 0, 0.9), box("vehicle", 20.6, 0.2, 0.7), box("pedestrian", 20.8, 0.3, 0.8)], [car], ("vehicle", "pedestrian"))
    assert sorted(kinds(r5["fp"])) == ["duplicate", "isolated"], r5                  # the pedestrian: the car is found, so a ghost; the second car: a duplicate
    # the counts equal the metric's matching on random worlds''')
# window test + mismatch refusals in the file section
rep('''        with open(os.path.join(run, "eval", "scene_records_n3.pkl"), "wb") as f:
            pickle.dump({"det": {"scene-1": {0.0: [(dets, gts), ([], [box("vehicle", 25, 2)])]}, "scene-2": {0.0: [(dets, gts)]}}}, f)
        rows, totals, nf = error_rows(run, 3)''', '''        outside = [box("vehicle", 10, 15, s=0.9), box("vehicle", 65, 0, s=0.9)]                    # 56 deg off the axis, and beyond 60 m: outside the scored window
        keys = ["tokscene-10", "tokscene-20", "tokscene-11"]
        with open(os.path.join(run, "eval", "scene_records_n3.pkl"), "wb") as f:
            pickle.dump({"det": {"scene-1": {0.0: [(dets + outside, gts + [box("vehicle", 10, -15), box("vehicle", 70, 0)]), ([], [box("vehicle", 25, 2)])]},
                                 "scene-2": {0.0: [(dets, gts)]}}}, f)
        rows, totals, nf = error_rows(run, 3)''')
rep('''        text = summaries(rows, totals)''', '''        assert not any(x["x"] in (10.0, 65.0, 70.0) for x in rows), "boxes outside the scored window are no errors"
        text = summaries(rows, totals)''')
rep('''        os.rename(os.path.join(run, "eval", "suite", "quant", "eval_n3"), os.path.join(run, "eval", "suite", "quant", "eval_curves_fusion_n3"))''', '''        with open(os.path.join(run, "eval", "scene_records_n3.pkl"), "rb") as f:                   # the keys of the records must be the store's: another order is refused
            rec = pickle.load(f)
        rec["meta"] = {"frames": keys}
        with open(os.path.join(run, "eval", "scene_records_n3.pkl"), "wb") as f:
            pickle.dump(rec, f)
        assert error_rows(run, 3)[2] == 3
        rec["meta"] = {"frames": keys[::-1]}
        with open(os.path.join(run, "eval", "scene_records_n3.pkl"), "wb") as f:
            pickle.dump(rec, f)
        try:
            error_rows(run, 3)
        except SystemExit:
            pass
        else:
            raise AssertionError("records of another frame order must be refused")
        rec["det"]["scene-2"][0.0].append(([], []))                                              # one more frame of boxes than the store lists
        with open(os.path.join(run, "eval", "scene_records_n3.pkl"), "wb") as f:
            pickle.dump(rec, f)
        try:
            error_rows(run, 3)
        except SystemExit:
            pass
        else:
            raise AssertionError("a count mismatch must be refused")
        os.rename(os.path.join(run, "eval", "suite", "quant", "eval_n3"), os.path.join(run, "eval", "suite", "quant", "eval_curves_fusion_n3"))''')
rep('''        assert len(load_frames(run, 3)) == 3, "the store name may carry the --curves / --fusion parts"''', '''        assert len(load_frames(run, 3)) == 3, "the store name may carry the --curves / --fusion parts"
        assert len(load_frames(run, 3, store="eval_curves_fusion_n3")) == 3 and find_store(run, 3, store="eval_curves_fusion_n3").endswith("eval_curves_fusion_n3")''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
