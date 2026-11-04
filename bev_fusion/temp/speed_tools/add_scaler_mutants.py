import re
p = "mutate_host.py"
s = open(p, encoding="utf-8").read()
s = s.replace('H, S = "test_host_overhead.py", "test_speed_flags.py"', 'H, S, G = "test_host_overhead.py", "test_speed_flags.py", "test_grad_scaler.py"')
s = s.replace("    for t in (H, S):\n", "    for t in (H, S, G):\n")
new = '''    ("BASELINE scaler (no change, must PASS)", None, None, None, G),
    ("scaler: gradients not unscaled before the clip", "training/train.py", "        scaler.unscale_(opt)\n    if before_clip", "        pass\n    if before_clip", G),
    ("scaler: an overflow does not lower the scale", "training/train.py", "scaler.update(scaler.get_scale() * scaler.get_backoff_factor())", "scaler.update(scaler.get_scale())", G),
    ("scaler: a norm-only overflow steps anyway", "training/train.py", "        opt.zero_grad(set_to_none=True)\n        if scaler is not None:\n            scaler.update(scaler.get_scale() * scaler.get_backoff_factor())", "        if scaler is not None:\n            scaler.step(opt)\n            scaler.update(scaler.get_scale() * scaler.get_backoff_factor())\n        opt.zero_grad(set_to_none=True)", G),
    ("scaler: an overflow keeps the gradients", "training/train.py", "        opt.zero_grad(set_to_none=True)\n        if scaler is not None:\n            scaler.update", "        if scaler is not None:\n            scaler.update", G),
    ("bad steps: an overflow always counts", "training/train.py", "scaler.get_scale() >= 1.0:\n        return bad", "scaler.get_scale() >= 1e9:\n        return bad", G),
    ("bad steps: an overflow never counts", "training/train.py", 'if status == "ok":\n        return 0', 'if status in ("ok", "overflow"):\n        return 0', G),
    ("run_clip: the loss is not scaled", "training/train.py", "(mean_loss if scaler is None else scaler.scale(mean_loss)).backward", "(mean_loss).backward", G),
    ("amp_context: fp16 runs in bfloat16", "training/train.py", '"fp16": torch.float16}', '"fp16": torch.bfloat16}', G),
    ("make_scaler: also for bf16", "training/train.py", 'if cfg.amp != "fp16":\n        return None', 'if cfg.amp == "off":\n        return None', G),
    ("checkpoint: the scaler state is not saved", "training/train.py", 'return {"scaler": scaler.state_dict()} if scaler is not None else {}', "return {}", G),
    ("amp_context: an unknown name is accepted", "training/train.py", 'if amp != "off" and amp not in AMP_DTYPES:', "if False:", G),
]
'''
s = s.replace("    (\"profiler: item counted twice again\"", "XXMARK", 1)
# insert the new mutants before the closing bracket of MUTANTS
i = s.index("XXMARK")
s = s.replace("XXMARK", "    (\"profiler: item counted twice again\"", 1)
j = s.index("]\n\n\ndef fresh")
s = s[:j] + new.rstrip("\n").rsplit("]", 1)[0] + s[j:]
open(p, "w", encoding="utf-8").write(s)
print("ok")
