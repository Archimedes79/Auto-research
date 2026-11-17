p = "training/train/jwire_train.py"
s = open(p, encoding="utf-8").read()
old = '''            assert "exists" in (refused(run_pipeline, 5, "e350_w_s5", pair_run="e324_s_s5") or "")
'''
new = '''            assert "exists" in (refused(run_pipeline, 5, "e350_w_s5", pair_run="e324_s_s5") or "")
            os.makedirs(os.path.join(d, "runs", "e350_w_s6"))
            open(os.path.join(d, "runs", "e350_w_s6", "wire_agreement.json"), "w").write("{}")                  # ... and a folder with only a wire_agreement.json
            assert "exists" in (refused(run_pipeline, 6, "e350_w_s6", pair_run="e324_s_s6") or "")
'''
assert s.count(old) == 1
open(p, "w", encoding="utf-8").write(s.replace(old, new))
