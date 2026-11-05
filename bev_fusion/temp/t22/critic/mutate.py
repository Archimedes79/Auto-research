import os, re, subprocess, sys
ROOT = "temp/t22/critic/mut"
PY = os.path.expanduser("~/miniconda3/envs/gcar/bin/python")
M = {
 "A inverted ida direction": ("student_encoder.py", "xy = (pts @ (nuscenes_front_ida(dst_hw=hw) @ torch.linalg.inv(bevfusion_ida())).T)[..., :2]", "xy = (pts @ (bevfusion_ida() @ torch.linalg.inv(nuscenes_front_ida(dst_hw=hw))).T)[..., :2]"),
 "B no half-cell offset (stride-1)/2 -> 0": ("student_encoder.py", "cell = (xy - (stride - 1) / 2) / stride", "cell = xy / stride"),
 "C align_corners False in decode": ("student_encoder.py", 'padding_mode="border", align_corners=True)', 'padding_mode="border", align_corners=False)'),
 "D KL direction reversed": ("distill_student.py", 'm["kl" + k] = (t * (t.clamp_min(1e-8).log() - out["logit" + k].float().log_softmax(1))).sum(1)', 'sp = out["logit" + k].float().softmax(1); m["kl" + k] = (sp * (out["logit" + k].float().log_softmax(1) - t.clamp_min(1e-8).log())).sum(1)'),
 "E validation with live weights, not EMA": ("distill_student.py", "        with ema.applied(model):\n            return validate(model, val_loader, device)", "        return validate(model, val_loader, device)"),
 "E2 final validation with live weights": ("distill_student.py", "    with ema.applied(model):\n        val = validate(model, val_loader, device)", "    val = validate(model, val_loader, device)"),
 "F samples/s without the batch factor": ("distill_student.py", 'return {"samples_per_s": self.size * len(self.durs) / sum(self.durs)', 'return {"samples_per_s": len(self.durs) / sum(self.durs)'),
 "G skip rule off by one in mean_rate": ("distill_student.py", "n = self.steps - self.skip", "n = self.steps"),
 "I H8 band rows without the warp rows": ("distill_student.py", "rows8 = h8_warp_grid()[1][:, 0, 1].numpy() if model.grid8 is not None else 8 * np.arange(32) + 3.5", "rows8 = 8 * np.arange(32) + 3.5"),
 "S padding zeros": ("student_encoder.py", 'padding_mode="border"', 'padding_mode="zeros"'),
 "X H8 target read for the wrong token": ("distill_student.py", "(p8, c8) = self.store16.get(tok), self.store8.get(tok)", "(p8, c8) = self.store16.get(tok), self.store8.get(self.tokens[(i + 1) % len(self.tokens)])"),
 "Y H16 target read for the wrong token": ("distill_student.py", "(p16, c16), (p8, c8) = self.store16.get(tok), self.store8.get(tok)", "(p16, c16), (p8, c8) = self.store16.get(self.tokens[(i + 1) % len(self.tokens)]), self.store8.get(tok)"),
 "Z val bands from H8 rows given H16 ys": ("distill_student.py", 'band_rows(ys["16" if name.endswith("16") else "8"])', 'band_rows(ys["16"])'),
 "W frozen BN eval removed": ("student_encoder.py", "        if self._frozen:\n            self.backbone.eval()", "        pass"),
 "V mean/std applied after float without std": ("student_encoder.py", "(img.float() - self.mean) / self.std", "(img.float() - self.mean)"),
 "U val mean over all cells incl. wrong denominators": ("distill_student.py", "per_row = (r / (n * width[name])).numpy()", "per_row = (r / (n * width[name] * 2)).numpy()"),
 "T drop_last/shuffle swapped on val": ("distill_student.py", "shuffle=train, drop_last=train", "shuffle=train, drop_last=False"),
 "R teacher not renormalised": ("distill_student.py", "        t = t / t.sum(1, keepdim=True)\n", ""),
 "Q lr_groups mult ignored": ("student_encoder.py", '{prefix + "backbone": {"lr": lr * backbone_mult}}', '{prefix + "backbone": {"lr": lr}}'),
 "P eval_every zero rule dropped": ("distill_student.py", "        cfg.eval_every = 0 ", "        pass "),
}
which = sys.argv[1:] or list(M)
for name in which:
    f, old, new = M[name]
    path = os.path.join(ROOT, "training", f)
    src = open(path, encoding="utf-8").read()
    if old not in src:
        print("NOT FOUND", name); continue
    open(path, "w", encoding="utf-8").write(src.replace(old, new, 1))
    try:
        r = subprocess.run([PY, ROOT + "/tests/test_student_encoder.py"], cwd=".", capture_output=True, text=True, timeout=300)
        out = r.stdout + r.stderr
        fails = [l.strip() for l in out.splitlines() if "FAIL" in l or "Traceback" in l or "Error" in l][:3]
        print(("KILLED  " if r.returncode != 0 else "SURVIVED"), name, "|", "; ".join(fails)[:200], flush=True)
    except subprocess.TimeoutExpired:
        print("TIMEOUT", name)
    finally:
        open(path, "w", encoding="utf-8").write(src)
