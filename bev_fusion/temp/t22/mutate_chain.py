"""scratch: plant bugs in a copy of the stage-B chain code and check that tests/test_student_chain.py (and for two of them tests/test_slots.py) fails for each; run in WSL from bev_fusion/"""
import os, shutil, subprocess, sys
SRC, MUT = os.path.abspath("."), os.path.abspath("temp/t22/mut_chain")
FC, SE, TR, MO = "feature_cache.py", "student_encoder.py", "train.py", "model.py"
M = [
 (FC, 'p = torch.cat([stud["prob" + h].float()[:, None], p[:, 1:]], 1)\n                    ctx = torch.cat([stud["ctx" + h].float()[:, None], ctx[:, 1:]], 1)',
      'p = torch.cat([p[:, :-1], stud["prob" + h].float()[:, None]], 1)\n                    ctx = torch.cat([ctx[:, :-1], stud["ctx" + h].float()[:, None]], 1)', "wrong frame replaced (the last)"),
 (FC, 'p = torch.cat([stud["prob" + h].float()[:, None], p[:, 1:]], 1)\n                    ctx = torch.cat([stud["ctx" + h].float()[:, None], ctx[:, 1:]], 1)',
      'p = stud["prob" + h].float()[:, None].expand_as(p)\n                    ctx = stud["ctx" + h].float()[:, None].expand_as(ctx)', "all frames replaced"),
 (FC, 'h = self.student.HEAD_OF[src.features]', 'h = {"dualbev": "8", "bevfusion": "16"}[src.features]', "sources swapped (p16 into BEVFusion's lift)"),
 (FC, '"t_p" + h: p[:, 0].detach(), "t_c" + h: ctx[:, 0].detach()}', '"t_p" + h: p[:, 0], "t_c" + h: ctx[:, 0]}', "teacher not detached"),
 (FC, '"t_p" + h: p[:, 0].detach()', '"t_p" + h: p[:, -1].detach()', "teacher side from the last frame"),
 (FC, 'return self.student(batch["lane_img"])', 'return self.student(batch["lane_img"].flip(0))', "student on the image of the wrong sample"),
 (FC, 'return self.student(batch["lane_img"])', 'return self.student(batch["lane_img"] * 0)', "student ignores the image (camera-off class)"),
 (FC, 'stud, side = (self._run_student(batch) if self.student is not None else None), {}', 'stud, side = None, {}', "option ignored by the cached model"),
 (FC, 'self._student_side = side or None', 'self._student_side = None', "student_side never set"),
 (SE, 'if self._bn_eval:', 'if False:', "BatchNorm in train mode under model.train()"),
 (SE, 'start * (1.0 - (1.0 - floor) * min(1.0, max(0.0, step_frac) / until))', 'start * (1.0 - (1.0 - floor) * min(1.0, max(0.0, step_frac) / 1.0))', "schedule: until ignored"),
 (SE, 'start * (1.0 - (1.0 - floor) * min(1.0, max(0.0, step_frac) / until))', 'start * (1.0 - floor * min(1.0, max(0.0, step_frac) / until))', "schedule: floor taken as the drop"),
 (SE, 'start * (1.0 - (1.0 - floor) * min(1.0, max(0.0, step_frac) / until))', 'start * (1.0 - (1.0 - floor) * max(0.0, step_frac) / until)', "schedule: no constant after until"),
 (SE, 'names[-1]) if names else None', 'names[0]) if names else None', "student_init takes the oldest checkpoint"),
 (SE, '    if fresh:\n', '    if False:\n', "student_init not strict"),
 (SE, '    fresh = warm_start(file, model, nn.Module(), ck=ck)', '    model.load_state_dict({k[6:]: v for k, v in ck["state"].items() if k.startswith("model.")}, strict=False)\n    fresh = []', "student_init loads the live state, not the EMA"),
 (TR, 'out["total"] = out["total"] + student_anchor_weight(step_frac, cfg.student_anchor, cfg.student_anchor_floor, cfg.student_anchor_until) * la',
      'out["total"] = out["total"] - student_anchor_weight(step_frac, cfg.student_anchor, cfg.student_anchor_floor, cfg.student_anchor_until) * la', "anchor sign"),
 (TR, 'out["student_anchor"] = la.detach()', 'out["student_anchor"] = la\n                out["total"] = out["total"] + la', "validation adds the anchor to the total"),
 (TR, 'out["student_anchor"] = la.detach()', 'out["student_anchor"] = la', "validation keeps the graph"),
 (TR, 'cfg is not None and cfg.student_anchor > 0:', 'cfg is not None and cfg.student_anchor >= 0:', "anchor computed at weight 0"),
 (TR, '"radar_kept", "student_anchor")', '"radar_kept")', "anchor not logged"),
 (TR, '"lr_groups", "student_anchor")', '"lr_groups")', "anchor not a resume switch"),
 (TR, 'and not cfg.student_trainable\n', 'and cfg.student_trainable\n', "_external wrong (trainable student external)"),
 (TR, '(("student.",) if frozen_student else ())', '()', "_external: frozen student in the checkpoints"),
 (TR, 'if cfg.student_anchor > 0 and not (mcfg.camera_student and mcfg.student_trainable):', 'if False:', "build: anchor without a trainable student accepted"),
 (TR, 'if mcfg.camera_student and not cfg.lane_images:', 'if False:', "build: no lane_images accepted"),
 (MO, 'if cfg.camera_features != "dual":', 'if False:', "check_config hole: any camera_features"),
 (MO, 'if cfg.camera_off:\n            errs.append("camera_student reads', 'if False:\n            errs.append("camera_student reads', "check_config hole: camera_off"),
 (MO, 'if not cfg.student_trainable and not cfg.student_init:', 'if False:', "check_config hole: frozen without init"),
 (MO, 'if cfg.student_frames == "all":', 'if False:', "student_frames all accepted"),
 (MO, 'if cfg.camera_student and not self.CACHED:', 'if False:', "base model accepts the option"),
 (MO, 'student.requires_grad_(cfg.student_trainable)', 'student.requires_grad_(True)', "frozen student requires grad"),
 (MO, 'student.keep_bn_eval()\n', '', "BatchNorm flag never set"),
 (MO, 'student_side, self._student_side = self._student_side, None', 'student_side = self._student_side', "student_side not cleared"),
 (MO, 'if student_side is not None:                              # t22: the anchor', 'if False:                              # t22: the anchor', "student_side not in the result"),
 (MO, 'prob0 = student_side["p16"].float() ', 'prob0 = batch["prob"][:, 0].float() ', "lane branch still on the cached depth"),
 (MO, 'prob0 = student_side["p16"].float() ', 'prob0 = batch["prob"][:, 1].float() ', "lane branch on the cached depth of the wrong frame"),
 (MO, 'prob0 = student_side["p16"].float() ', 'prob0 = student_side["p8"].float() ', "lane branch on the BEVFusion depth (wrong head)"),
 (MO, 'prob0 = student_side["p16"].float() ', 'prob0 = student_side["p16"].detach().float() ', "lane branch depth without the student's gradient"),
]
want = sys.argv[1:]
if want:
    M = [x for x in M if any(w in x[3] for w in want)]
def fresh():
    if os.path.exists(MUT): shutil.rmtree(MUT)
    os.makedirs(MUT)
    ign = shutil.ignore_patterns("__pycache__", "configs", "legacy", "*.pt")
    shutil.copytree(os.path.join(SRC, "training"), os.path.join(MUT, "training"), ignore=ign)
    shutil.copytree(os.path.join(SRC, "evaluation"), os.path.join(MUT, "evaluation"), ignore=ign)
    shutil.copytree(os.path.join(SRC, "tests"), os.path.join(MUT, "tests"), ignore=ign)
    os.makedirs(os.path.join(MUT, "temp"))
env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
def run(test="tests/test_student_chain.py"):
    r = subprocess.run([sys.executable, test], cwd=MUT, capture_output=True, text=True, env=env)
    out = [l.strip() for l in r.stdout.splitlines() if "FAIL" in l][:2]
    if r.returncode and not out and r.stderr.strip():
        out = [r.stderr.strip().splitlines()[-1]]
    return r.returncode, out
fresh(); rc, _ = run(); print("baseline exit", rc, flush=True)
alive = []
for f, old, new, name in M:
    fresh()
    p = os.path.join(MUT, "training", f); s = open(p, encoding="utf-8").read()
    if old not in s: print("PATTERN MISSING:", name, flush=True); alive.append(name + " (pattern missing)"); continue
    open(p, "w", encoding="utf-8", newline="\n").write(s.replace(old, new, 1))
    rc, why = run()
    print(("killed " if rc else "ALIVE  ") + name, "|", (why[0] if why else "")[:160], flush=True)
    if not rc: alive.append(name)
print("alive:", alive)
