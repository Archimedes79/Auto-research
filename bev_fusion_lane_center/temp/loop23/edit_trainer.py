p = "training/trainer.py"
s = open(p, encoding="utf-8").read()
Q = '"' * 3


def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (s.count(a), a[:70])
    s = s.replace(a, b)


# header
rep("  * periodic `eval_fn(model, step) -> dict` written to status.json\n" + Q + "\n",
    "  * periodic `eval_fn(model, step) -> dict` written to status.json\n"
    "  * the run-directory contract of `autoresearch/trainsuite` (`python -m trainsuite status / health / curves / stop / watch RUN`): `config.json` ({\"steps\": N, ...}), `log.jsonl` (one row per\n"
    "    logged step: step, t, total, lr, grad_norm, s_per_step, gpu_gb, the scalars of the loss's info dict, and with `probe=True` the keys grad.<module> / upd.<module> / t_data / t_step; an eval\n"
    "    row {\"step\", \"val\": {...}}), `train.log` (the text output, also `log.txt`), a stop request: the file CONTROL = {\"cmd\": \"stop\"} is looked at before every step, then the trainer saves a\n"
    "    checkpoint, writes STOPPED and exits with code 75 (never 0, so no `.done` and nobody evaluates a half-trained model); the files of an earlier stop are moved aside at the start;\n"
    "    an `incident_<step>.json` for a skipped non-finite step\n" + Q + "\n")
rep("import traceback\n", "import traceback\n")
rep("import torch\n\nRUNS = BP.RUNS\n",
    "import torch\n\nimport probe as PR  # noqa: E402  (training/probe.py: gradient and update size per module, copied from the BEV system)\n\n"
    "RUNS = BP.RUNS\n"
    "EXIT_STOPPED = 75                       # a stopped run exits with this code and never with 0 (`.done` means \"finished correctly\"; autoresearch/tools/run.py knows the code)\n")
rep('        with open(os.path.join(self.dir, "log.txt"), "a", encoding="utf-8") as f:\n            f.write(line + "\\n")\n',
    '        for name in ("log.txt", "train.log"):                              # log.txt: the name of every older run, train.log: the trainsuite contract\n'
    '            with open(os.path.join(self.dir, name), "a", encoding="utf-8") as f:\n'
    '                f.write(line + "\\n")\n\n'
    '    def jsonl(self, row):\n'
    '        ' + Q + 'one row of log.jsonl (the metrics stream the trainsuite reads); a reader on the Windows side may hold the file open: a failed write is printed, not raised' + Q + '\n'
    '        try:\n'
    '            with open(os.path.join(self.dir, "log.jsonl"), "a", encoding="utf-8") as f:\n'
    '                f.write(json.dumps(row, default=float) + "\\n")\n'
    '        except OSError as e:\n'
    '            print(f"WARNING: log.jsonl row not written ({e})", flush=True)\n\n'
    '    def stop_requested(self):\n'
    '        ' + Q + 'the reason (a string) when the file CONTROL says {"cmd": "stop"}, else None (same reading as autoresearch/trainsuite/control.py)' + Q + '\n'
    '        try:\n'
    '            with open(self.path("CONTROL"), encoding="utf-8") as fh:\n'
    '                d = json.load(fh)\n'
    '            return str(d.get("reason", "")) if d.get("cmd") == "stop" else None\n'
    '        except (OSError, ValueError, AttributeError):\n'
    '            return None\n\n'
    '    def clear_control(self):\n'
    '        ' + Q + 'at the start: CONTROL / STOPPED of an earlier stop are moved aside (*.old), so a new process never obeys an old request' + Q + '\n'
    '        for n in ("CONTROL", "STOPPED"):\n'
    '            if os.path.exists(self.path(n)):\n'
    '                os.replace(self.path(n), self.path(n) + ".old")\n')
rep('device="cuda", args=None, params=None, max_bad=20):', 'device="cuda", args=None, params=None, max_bad=20, probe=False, probe_depth=1, config=None):')
rep("    bad = 0                                                              # non-finite steps in a row\n",
    "    bad = 0                                                              # non-finite steps in a row\n"
    "    run.clear_control()\n"
    '    with open(run.path("config.json"), "w", encoding="utf-8") as fh:        # the trainsuite reads `steps` from it (percent, ETA)\n'
    '        json.dump(dict(config or {}, steps=steps, lr=lr, amp=amp, clip=clip, seed=seed, args=args), fh, indent=1, default=str)\n'
    "    mprobe, timer = (PR.ModuleProbe(model, every=log_every, depth=probe_depth), PR.StepTimer()) if probe else (None, None)\n"
    '    last_info, gn_last, prow = {}, float("nan"), {}\n')
rep("        nonlocal n_skip, bad, step\n        n_skip += 1\n",
    '        nonlocal n_skip, bad, step\n        PR.dump_incident(run.dir, step + 1, "nonfinite_" + what, losses=last_info)\n        n_skip += 1\n')
rep("        while step < steps:\n            batch = batch_fn(step) if batch_fn else next(it)\n",
    "        while step < steps:\n"
    "            reason = run.stop_requested()                                # `python -m trainsuite stop <run>`\n"
    "            if reason is not None:\n"
    "                save_ckpt()\n"
    '                with open(run.path("STOPPED"), "w", encoding="utf-8") as fh:\n'
    '                    json.dump({"step": step, "reason": reason, "t": time.time()}, fh)\n'
    '                run.status(state="stopped", step=step)\n'
    '                run.log(f"step {step}: stop requested ({reason}), checkpoint saved, exit {EXIT_STOPPED}")\n'
    "                raise SystemExit(EXIT_STOPPED)\n"
    "            if timer is not None:\n"
    "                with timer.data():\n"
    "                    batch = batch_fn(step) if batch_fn else next(it)\n"
    "            else:\n"
    "                batch = batch_fn(step) if batch_fn else next(it)\n")
rep("                loss, info = loss_fn(model, batch)\n",
    "                loss, info = loss_fn(model, batch)\n"
    "            last_info = info or {}\n"
    "            sp = mprobe.begin(step + 1) if mprobe is not None else None         # a StepProbe on the logged steps (the count after this step is a multiple of log_every)\n")
rep('                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), clip if clip else float("inf"))     # (inf: only measures)\n',
    "                if sp is not None:\n"
    "                    sp.grads()                                           # gradient norm per module BEFORE clipping (and a copy of the weights for the update ratio)\n"
    '                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), clip if clip else float("inf"))     # (inf: only measures)\n'
    "                gn_last = float(gn)\n")
rep("                        bad = 0\n                        step += 1\n                        hist.append(float(loss))\n                        updated = True\n",
    "                        bad = 0\n                        step += 1\n                        hist.append(float(loss))\n                        updated = True\n"
    "                        prow = sp.finish() if sp is not None else {}\n")
rep("                if guard and step >= 4 * log_every and os.path.exists(ck):\n",
    "                row = dict(step=step, t=time.time(), total=cur, lr=opt.param_groups[0][\"lr\"], grad_norm=gn_last, s_per_step=(time.time() - t0) / max(step - step0, 1),\n"
    "                           gpu_gb=(torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() and device == \"cuda\" else 0.0))\n"
    "                row.update({k: float(v) for k, v in (info or {}).items() if isinstance(v, (int, float)) or hasattr(v, \"item\")})\n"
    "                row.update(prow)\n"
    "                if timer is not None:\n"
    "                    row.update(timer.row())\n"
    "                run.jsonl(row)\n"
    "                if guard and step >= 4 * log_every and os.path.exists(ck):\n")
rep("    t0 = time.time()\n    model.train()\n    try:", "    t0, step0 = time.time(), step\n    model.train()\n    try:")
rep('                run.status(**{f"eval_{step}": res})\n',
    '                run.status(**{f"eval_{step}": res})\n'
    '                run.jsonl({"step": step, "val": {k: float(v) for k, v in (res or {}).items() if isinstance(v, (int, float))}})\n')
rep("    except BaseException as e:  # keep the checkpoint, tell the next start what happened\n",
    "    except SystemExit:                                                   # a stop request (exit 75): not a failure, the next start resumes\n        raise\n"
    "    except BaseException as e:  # keep the checkpoint, tell the next start what happened\n")
open(p, "w", encoding="utf-8").write(s)
print("ok")
