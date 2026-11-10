"""scratch: plant bugs in a copy of the student code and check that tests/test_student_encoder.py fails for each (run in WSL from bev_fusion/)"""
import os, shutil, subprocess, sys
SRC, MUT = os.path.abspath("."), os.path.abspath("temp/t22/mut")
M = [
 ("student_encoder.py", 'padding_mode="border"', 'padding_mode="zeros"', "warp pads with zeros"),
 ("student_encoder.py", 'padding_mode="border", align_corners=True)', 'padding_mode="border", align_corners=False)', "decode align_corners False"),
 ("student_encoder.py", "        if self._frozen:\n            self.backbone.eval()", "        if self._frozen:\n            pass", "frozen BN not in eval"),
 ("student_encoder.py", "lr * backbone_mult", "lr", "backbone mult ignored"),
 ("student_encoder.py", "p.requires_grad_(not flag)", "p.requires_grad_(True)", "freeze does not freeze"),
 ("student_encoder.py", "o16 = self.head16(self.smooth[1](p16))", "o16 = self.head16(F.avg_pool2d(self.smooth[0](p8), 2))", "H16 from the stride-8 map"),
 ("student_encoder.py", "cell = (xy - (stride - 1) / 2) / stride", "cell = xy / stride", "warp assumes cell corners"),
 ("student_encoder.py", "(nuscenes_front_ida(dst_hw=hw) @ torch.linalg.inv(bevfusion_ida())).T", "(bevfusion_ida() @ torch.linalg.inv(nuscenes_front_ida(dst_hw=hw))).T", "ida order reversed"),
 ("student_encoder.py", "self.register_buffer(\"grid8\", h8_warp_grid()[0] if warp_h8 else None, persistent=False)", "self.register_buffer(\"grid8\", None, persistent=False)", "no warp built"),
 ("student_encoder.py", "return self.backbone((img.float() - self.mean) / self.std)", "return self.backbone(img.float() / 255)", "no normalisation"),
 ("distill_student.py", "m[\"kl\" + k] = (t * (t.clamp_min(1e-8).log() - out[\"logit\" + k].float().log_softmax(1))).sum(1)", "lp = out[\"logit\" + k].float().log_softmax(1)\n        m[\"kl\" + k] = (lp.exp() * (lp - t.clamp_min(1e-8).log())).sum(1)", "KL reversed"),
 ("distill_student.py", "        t = t / t.sum(1, keepdim=True)\n", "", "teacher not renormalised"),
 ("distill_student.py", 'band_rows(ys["16" if name.endswith("16") else "8"])', 'band_rows(ys["16"])', "H8 bands from the H16 rows"),
 ("distill_student.py", "per_row = (r / (n * width[name])).numpy()", "per_row = (r / n).numpy()", "row mean without the columns"),
 ("distill_student.py", "if ckpts and not a.resume:", "if False:", "no --resume guard"),
 ("distill_student.py", " and os.path.exists(image_path(img_root, t)))", ")", "sweep frames kept as key frames"),
 ("distill_student.py", "ram_mb < MIN_RAM_MB", "ram_mb < 0", "RAM gate off"),
 ("distill_student.py", "gpu_mb is None or gpu_mb > MAX_GPU_MB", "gpu_mb is None", "GPU limit off"),
 ("distill_student.py", "stop_check=stop_check)", ")", "stop_check not passed to fit"),
 ("distill_student.py", "cfg.lr_groups = model.lr_groups(cfg.lr, cfg.backbone_mult)", "pass", "lr_groups not set (A2 would run the backbone at full lr)"),
 ("distill_student.py", "model.freeze_backbone(cfg.freeze_backbone)", "pass", "--freeze-backbone ignored"),
 # follow-up of the critic (2026-11-05)
 ("distill_student.py", "(p16, c16), (p8, c8) = self.store16.get(tok), self.store8.get(tok)", "(p16, c16), (p8, c8) = self.store16.get(tok), self.store8.get(self.tokens[(i + 1) % len(self.tokens)])", "H8 target read for the next token"),
 ("distill_student.py", "    with ema.applied(model):\n        val, curves = score(model, val_loader, device, null)", "    if True:\n        val, curves = score(model, val_loader, device, null)", "final validation with live weights"),
 ("distill_student.py", "\"samples_per_s\": self.size * len(self.durs) / sum(self.durs)", "\"samples_per_s\": len(self.durs) / sum(self.durs)", "samples/s without the batch"),
 ("distill_student.py", "        n = self.steps - self.skip\n", "        n = self.steps - self.skip + 1\n", "mean_rate off by one"),
 ("distill_student.py", "        if self.steps > self.skip:", "        if self.steps >= self.skip:", "skip off by one"),
 ("distill_student.py", "drop_last=train", "drop_last=False", "drop_last off on train"),
 ("distill_student.py", "shuffle=train", "shuffle=False", "no shuffle on train"),
 ("distill_student.py", "if cfg.steps <= cfg.eval_every:", "if cfg.steps < cfg.eval_every:", "eval_every rule off by one"),
 ("distill_student.py", "p = null[k].clamp_min(NULL_FLOOR)", "p = null[k]", "null without floor"),
 ("distill_student.py", "ds.tokens[:n]", "ds.tokens", "null over all frames"),
 ("distill_student.py", "            if k[0] == \"p\":\n                v = v / v.sum(1, keepdim=True)\n", "", "null without renormalising the frames"),
 ("distill_student.py", "return 1 - kl / kl_null if kl_null > 0", "return 1 - kl_null / kl if kl_null > 0", "skill inverted"),
 ("distill_student.py", "per_col = (self.cols[name] / (self.n * len(self.rows[name]))).numpy()", "per_col = (self.cols[name] / self.n).numpy()", "per-column mean wrong divisor"),
 ("distill_student.py", "per_row = (self.rows[name] / (self.n * len(self.cols[name]))).numpy()", "per_row = (self.rows[name] / self.n).numpy()", "per-row mean wrong divisor"),
 ("distill_student.py", "    warm_start(path, model, torch.nn.Module(), ck=ck)", "    model.load_state_dict({k[6:]: v for k, v in ck[\"state\"].items() if k.startswith(\"model.\")}, strict=False)", "eval-only loads live weights"),
 ("distill_student.py", "os.path.join(out_dir, \"eval_metrics.json\")", "os.path.join(out_dir, \"metrics.json\")", "eval-only overwrites metrics.json"),
 ("distill_student.py", "    path = os.path.join(out_dir, ckpts[-1])", "    path = os.path.join(out_dir, ckpts[0])", "eval-only takes the oldest checkpoint"),
 ("distill_student.py", "    if a.device.startswith(\"cuda\") and not torch.cuda.is_available():", "    if False:", "missing CUDA not refused"),
 ("distill_student.py", "        torch.set_num_threads(a.threads)\n", "", "--threads ignored"),
]
if len(sys.argv) > 1:
    M = [x for x in M if x[3] in sys.argv[1:]]
def fresh():
    if os.path.exists(MUT): shutil.rmtree(MUT)
    os.makedirs(MUT)
    shutil.copytree(os.path.join(SRC, "training"), os.path.join(MUT, "training"), ignore=shutil.ignore_patterns("__pycache__", "configs", "legacy"))
    os.makedirs(os.path.join(MUT, "tests")); shutil.copy(os.path.join(SRC, "tests", "test_student_encoder.py"), os.path.join(MUT, "tests"))
    os.symlink(os.path.join(SRC, "models"), os.path.join(MUT, "models"))
env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
def run():
    r = subprocess.run([sys.executable, "tests/test_student_encoder.py"], cwd=MUT, capture_output=True, text=True, env=env)
    return r.returncode, [l for l in r.stdout.splitlines() if "FAIL" in l][:2] + ([r.stderr.strip().splitlines()[-1]] if r.returncode and not any("FAIL" in l for l in r.stdout.splitlines()) else [])
fresh(); rc, _ = run(); print("baseline exit", rc, flush=True)
alive = []
for f, old, new, name in M:
    fresh()
    p = os.path.join(MUT, "training", f); s = open(p).read()
    if old not in s: print("PATTERN MISSING:", name); alive.append(name); continue
    open(p, "w").write(s.replace(old, new, 1))
    rc, why = run()
    print(("killed " if rc else "ALIVE  ") + name, "|", (why[0] if why else "")[:150], flush=True)
    if not rc: alive.append(name)
print("alive:", alive)

