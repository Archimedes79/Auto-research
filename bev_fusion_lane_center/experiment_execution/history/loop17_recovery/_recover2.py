"""Rebuild the files that a rollback removed, second version: Write / Edit tool calls (key 'new_str' accepted) plus the node / heredoc patches of the Bash commands that changed them (listed and reviewed by hand).
Output: a shadow tree (staging dir / relative path); nothing is copied into place. Usage: _recover2.py <transcript.jsonl> <staging dir>"""
import json
import os
import re
import sys

src, stage = sys.argv[1], sys.argv[2]
P = "C:\\ML-Experiment\\"
calls, results = [], {}
with open(src, encoding="utf-8", errors="replace") as f:
    for ln, line in enumerate(f):
        try:
            d = json.loads(line)
        except Exception:
            continue
        m = d.get("message")
        if not isinstance(m, dict):
            continue
        c = m.get("content")
        if not isinstance(c, list):
            continue
        for b in c:
            if not isinstance(b, dict):
                continue
            if m.get("role") == "assistant" and b.get("type") == "tool_use":
                calls.append((ln, b))
            elif m.get("role") == "user" and b.get("type") == "tool_result":
                results[b.get("tool_use_id")] = b
files, log = {}, []


def rep(path, a, b, tag):
    k = P + path.replace("/", "\\")
    if k not in files:
        log.append((tag, k, "PATCH TARGET MISSING"))
        return
    if a not in files[k]:
        log.append((tag, k, "PATCH OLD STRING NOT FOUND " + a[:50].replace("\n", "|")))
        return
    files[k] = files[k].replace(a, b, 1)
    log.append((tag, k, "ok"))


def append(path, text, tag):
    k = P + path.replace("/", "\\")
    if k not in files:
        log.append((tag, k, "APPEND TARGET MISSING"))
        return
    files[k] += text
    log.append((tag, k, "ok append"))


def heredoc(cmd, name, tag="PYEOF"):
    m = re.search(r"cat >> " + re.escape(name) + r" <<'" + tag + r"'\n(.*?)\n" + tag + r"\n", cmd, re.S)
    return (m.group(1) + "\n") if m else None


def patch(ln, cmd):
    if ln == 3380:
        a1 = '    ap.add_argument("--prev", type=int, default=1)'
        rep("scripts/p17_route.py", a1, a1 + '\n    ap.add_argument("--tag_suffix", default="", help="appended to the result file name")', ln)
        rep("scripts/p17_route.py", 'open(f"{E17}/{a.out}/route_{tag}.json", "w")', 'open(f"{E17}/{a.out}/route_{tag}{a.tag_suffix}.json", "w")', ln)
    elif ln == 3745:
        h = heredoc(cmd, "p17_mapnet.py")
        append("scripts/p17_mapnet.py", h, ln) if h else log.append((ln, "p17_mapnet.py", "HEREDOC NOT PARSED"))
        a = '    ap.add_argument("--seed", type=int, default=0)\n'
        rep("scripts/p17_mapft.py", a, a + '    ap.add_argument("--real_fail", type=int, default=0, help="1: half of the frames are real failure frames of the TRAIN rows (seven types, p17_dense_train.py)")\n    ap.add_argument("--share", type=float, default=0.5)\n    ap.add_argument("--init_ckpt", default=None, help="start from the map-conditioned weights of this run (47 input channels)")\n', ln)
        rep("scripts/p17_mapft.py", "    MN.widen(net)\n    net.train()", '    MN.widen(net)\n    if a.init_ckpt:\n        net.load_state_dict(torch.load(f"{C.OUT}/e123/{a.init_ckpt}/model.pt", map_location=dev), strict=False)\n    net.train()', ln)
        rep("scripts/p17_mapft.py", "    dl = torch.utils.data.DataLoader(ds, batch_size=PT.BS, shuffle=True,", '    if a.real_fail:\n        fail = MN.MapWrap(MN.RealFail(base, ["dark", "blur", "cam_front3_off", "fog", "radar_off", "extr3", "cam_front_off"]), "train", P, train=True, p_none=a.p_none, p_wrong=a.p_wrong, p_degrade=0.0, seed=a.seed + 1)\n        ds = MN.Mix2(ds, fail, a.share, n=iters * PT.BS + 1000, seed=a.seed)\n    dl = torch.utils.data.DataLoader(ds, batch_size=PT.BS, shuffle=not a.real_fail,', ln)
    elif ln == 4082:
        rep("scripts/p17_route.py", '    ap.add_argument("--tag_suffix"', '    ap.add_argument("--mapnet", type=int, default=0, help="1: --ckpt is a map-conditioned builder (p17_mapnet.py)")\n    ap.add_argument("--map_mode", default="exact")\n    ap.add_argument("--map_sigma", type=float, default=0.0)\n    ap.add_argument("--tag_suffix"', ln)
        rep("scripts/p17_route.py", "    if a.ckpt:\n        net.load_state_dict(", "    if a.mapnet:\n        import p17_mapnet as MN\n\n        MN.widen(net)\n        a.prev = 0\n    if a.ckpt:\n        net.load_state_dict(", ln)
        rep("scripts/p17_route.py", "        ds.P17 = ds.P17\n", '        if a.mapnet:\n            ds = MN.MapWrap(ds, "val", ds.P17, train=False, mode=a.map_mode, sigma=a.map_sigma)\n', ln)
    elif ln == 4280:
        rep("scripts/p17_master.py", '        for mode in ("m0", "m2", "mnone", "mwrong"):\n            if os.path.exists(f"{E17}/e140_gateB/{p}_gateB_{mode}.pkl"):', '        for mode in ("m0", "m1", "m2", "mnone", "mwrong"):\n            if os.path.exists(f"{E17}/e140_gateB/{p}_gateB_{mode}.pkl"):', ln)
    elif ln == 4364:
        h = heredoc(cmd, "loop17_text_e139.py")
        append("scripts/loop17_text_e139.py", h, ln) if h else log.append((ln, "loop17_text_e139.py", "HEREDOC NOT PARSED"))
    elif ln == 5158:
        rep("scripts/build_loop17_report.py", 'return next((v["route_success"] for k, v in c.items() if k.startswith(f"sigma{sg}_") and k.endswith(f"_route_{rt}_{ph}")), None)', 'return next((v["route_success"] for k, v in c.items() if k.startswith(f"sigma{sg}_") and k.endswith(f"_route_{rt}_{ph}")), float("nan"))', ln)
        rep("scripts/p17_extra.py", '(f"e135_route/{p}_single", f"e135_route/{p}_maxdecay_0.95", f"{p}_single", f"{p}: hold - single frame (champion; hold = recursion off)"),', '(f"e135_route/{p}_maxdecay_0.95", f"e135_route/{p}_single", f"{p}_maxdecay_0.95", f"{p}: hold - single frame (champion; hold = recursion off)"),', ln)


PATCH_LINES = {3380, 3745, 4082, 4280, 4364, 5158}
for ln, b in calls:
    nm, inp = b.get("name"), b.get("input", {})
    fp = inp.get("file_path")
    if nm in ("Write", "Edit") and fp and "ML-Experiment" in fp:
        r = results.get(b["id"])
        err = bool(r and r.get("is_error"))
        text = ""
        if r:
            cc = r.get("content")
            text = cc if isinstance(cc, str) else " ".join(x.get("text", "") for x in cc if isinstance(x, dict)) if isinstance(cc, list) else ""
        ok = (not err) and not text.startswith("<tool_use_error>")
        if nm == "Write":
            if ok:
                files[fp] = inp["content"]
            log.append((ln, fp, "Write ok" if ok else "Write FAILED"))
        else:
            if not ok:
                log.append((ln, fp, "Edit failed in the original (skipped)"))
                continue
            old = inp.get("old_string")
            new = inp.get("new_string", inp.get("new_str"))
            if old is None or new is None:
                log.append((ln, fp, "Edit UNKNOWN SHAPE " + ",".join(inp.keys())))
                continue
            if fp not in files:
                log.append((ln, fp, "Edit NO BASE"))
                continue
            if old not in files[fp]:
                log.append((ln, fp, "Edit OLD NOT FOUND (out of sync)"))
                continue
            files[fp] = files[fp].replace(old, new) if inp.get("replace_all") else files[fp].replace(old, new, 1)
            log.append((ln, fp, "Edit ok"))
    elif nm == "Bash" and ln in PATCH_LINES:
        patch(ln, inp.get("command", ""))
bad = [x for x in log if not str(x[2]).endswith("ok") and "Write ok" not in str(x[2]) and "Edit ok" not in str(x[2]) and "skipped" not in str(x[2])]
print("steps:", len(log), "problems:", len(bad))
for x in bad:
    print("  PROBLEM", x)
for x in log:
    if "skipped" in str(x[2]):
        print("  (original failed edit, skipped as in the original)", x)
n = 0
for fp, content in files.items():
    rel = fp.replace(P, "").replace("\\", "/")
    out = os.path.join(stage, rel)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w", encoding="utf-8", newline="").write(content)
    n += 1
print("files rebuilt:", n, "in", stage)
