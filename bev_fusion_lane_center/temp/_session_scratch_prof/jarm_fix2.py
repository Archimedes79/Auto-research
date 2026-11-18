p = "tools/jarm.py"
t = open(p, encoding="utf-8").read()
reps = [
('''    if not os.path.exists(path):
        return False
    with open(path, "rb") as fh:
        fh.seek(-1, 2)
        return fh.read(1) != b"."''','''    if not os.path.exists(path):
        return False
    if os.path.getsize(path) == 0:
        return True
    with open(path, "rb") as fh:
        fh.seek(-1, 2)
        return fh.read(1) != b"."'''),
('''[self.out("token_cell_perclass*.json")], False, 20, 0)''','''[self.out("token_cell_perclass_train_every3.json")], False, 20, 0)'''),
('''            if os.path.exists(log) and any(w in open(log, encoding="utf-8", errors="replace").read().lower() for w in ("loss nan", "loss inf", "not finite")):
                self.say("STOPPED: a non-finite loss (the card's kill)")
                return False
            self.launch("", f"{PYR} {self.trainer} --name {self.run} --seed {self.seed} --head-seed {self.head_seed} {DEFAULTS} {self.extra}", True, 90)
            if self.dry:
                return True''','''            if os.path.exists(log) and "non-finite" in open(log, encoding="utf-8", errors="replace").read():      # the trainer's line `step N: non-finite <loss|gradient> skipped`
                self.say("STOPPED: a non-finite loss or gradient in the log (the card's kill)")
                return False
            rc = self.launch("", f"{PYR} {self.trainer} --name {self.run} --seed {self.seed} --head-seed {self.head_seed} {DEFAULTS} {self.extra}", True, 90)
            if self.dry:
                return True
            if rc in (2, 3):                                                       # run.py refused the launch (a card, a hold, a slot rule) or it is blocked: a relaunch would be refused again
                self.say(f"STOPPED: run.py exit {rc} (refused or blocked)")
                return False'''),
('''relaunched (resume) up to 6 times; a non-finite loss stops it;''','''relaunched (resume) up to 6 times; a `non-finite` line in the trainer's log (the card's kill) or a refusal of run.py (exit 2 / 3) stops it;'''),
('''        self.run = f"{eid.lower()}_{arm}"
''','''        assert not arm.startswith(("vs_", "cell", "d115", "dump", "sweep", "choose")), f"arm name {arm}: it would make run ids of another arm's steps"
        self.run = f"{eid.lower()}_{arm}"
'''),
('''            table(["ref", "x"], D115)
            p = os.path.join(d, "x", "token_val.pkl")''','''            import contextlib
            import io
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                table(["ref", "x"], D115)
            assert "-0.0100" in buf.getvalue() and "x - ref" in buf.getvalue() and "873 frames" in buf.getvalue(), buf.getvalue()
            os.makedirs(os.path.join(d, "y"))                                           # a run with only a paired card of the same decode: the numbers come from it
            json.dump(dict(split="val", frames=873, metrics={k: 0.7 for k in METRICS}), open(card_path("y", D115, "ref"), "w"))
            assert metrics("y", D115)["topo_F1"] == 0.7
            assert os.path.basename(card_path("y", "0.3,0.3,0.5,on", "ref")) == "token_card_rec_vs_ref_rc0.3_0.2_0.6_off.json"
            p = os.path.join(d, "x", "token_val.pkl")'''),
('''            open(p, "wb").write(b"abc.")
            assert not dump_cut(p)''','''            open(p, "wb").write(b"abc.")
            assert not dump_cut(p)
            open(p, "wb").write(b"")
            assert dump_cut(p), "an empty dump is cut"'''),
]
for a, b in reps:
    assert t.count(a) == 1, a[:70]
    t = t.replace(a, b)
open(p, "w", encoding="utf-8").write(t)
print("ok")
