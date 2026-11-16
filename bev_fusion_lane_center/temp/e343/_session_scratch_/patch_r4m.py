import os
R = os.environ["BFLC_ROOT"]
src = open(os.path.join(R, "evaluation", "jtest_eval.py"), encoding="utf-8").read().split("\n")
M = os.path.join(R, "temp", "e343", "_session_scratch_", "mutate.py")
text = open(M, encoding="utf-8").read()


def line(prefix, contains=""):
    hits = [ln for ln in src if ln.startswith(prefix) and contains in ln]
    assert len(hits) == 1, (prefix, contains, len(hits))
    return hits[0]


rows = []


def add(name, old, new):
    rows.append((name, old, new))


add("pin card not required on TEST", '_require(bool(a.expect_code_card), "TEST: --expect-code-card is required', '_require(True, "TEST: --expect-code-card is required')
add("pin typed digest skipped when empty", "if want is not None and not (", "if want and not (")
add("pin split not checked", 'card.get("split") == "val" and card.get("frames") == EXPECT_FRAMES["val"]', 'card.get("frames") == EXPECT_FRAMES["val"]')
add("pin frames not checked", 'card.get("split") == "val" and card.get("frames") == EXPECT_FRAMES["val"]', 'card.get("split") == "val"')
add("pin run prefix not checked", 'str(card.get("run", "")).startswith(run_prefix)', "True")
add("pin rehearsal ok not checked", 'isinstance(rc, dict) and rc.get("ok") is True', "isinstance(rc, dict)")
add("pin drift not checked", '"code_drift" in card and not card["code_drift"]', "True")
add("pin digest of file list not checked", 'code_digest(files) == card.get("code_digest")', "True")
add("pin environment record not checked", 'isinstance(card.get("environment"), dict)', "True")
add("pin card digest ignored", 'if ref and ref["code_digest"] != digest:', "if False:")
add("environment compare off", "        if diff:\n            sys.exit(f\"STOP: the environment differs", "        if False:\n            sys.exit(f\"STOP: the environment differs")
add("environment_diff versions ignored", 'for sec in ("files", "versions")', 'for sec in ("files",)')
add("undigested not computed", "undigested = sorted(repo_modules() - loaded0 - set(code0))", "undigested = []")
add("undigested includes preloaded", "undigested = sorted(repo_modules() - loaded0 - set(code0))", "undigested = sorted(repo_modules() - set(code0))")
add("repo_modules temp kept", '"/temp/" not in "/" + rel', "True")
add("repo_modules GtGenerator kept", 'rel.split("/")[0] != "GtGenerator"', "True")
add("repo_modules relative kept", "not os.path.isabs(f) or ", "")
b_gate = line("    prov = provenance(run, device)", "STARTS") + "\n    if device != \"cpu\":"
add("gpu gate always (dump_b)", line("    prov = provenance(run, device)", "STARTS") + '\n    if device != "cpu":', line("    prov = provenance(run, device)", "STARTS") + '\n    if True:')
add("gpu gate removed (dump_b)", '        import gpu_gate\n        gpu_gate.wait_for_room(who=f"jtest_eval dump {run}")\n    write_head(out, dict(split=split, frames=None), prov)', '        pass\n    write_head(out, dict(split=split, frames=None), prov)')
add("gpu gate removed (dump_ends)", '        import gpu_gate\n        gpu_gate.wait_for_room(who=f"jtest_eval dump {run}")\n    write_head(out, dict(run=run, split=split, frames=None), prov)', '        pass\n    write_head(out, dict(run=run, split=split, frames=None), prov)')
add("gpu gate always (dump_ends)", line("    prov = provenance(run, device)", "at the start") + '\n    if device != "cpu":', line("    prov = provenance(run, device)", "at the start") + '\n    if True:')
add("TEST artefact rule off", "_require(rel in E321_SHA256, ", "_require(True, ")
add("precheck rows not validated", line("        load_dump(path, a.split, what, a.max)"), "        pass")
add("early control off", "early = {s: control_early(a, setup, s, frames, pre[s]) for s in a.seeds}", "early = {s: None for s in a.seeds}")
add("early counts scored again", 'sc = score_dumps(frames, dumps, ends, [v for v in variants if not (early and v["name"] in early["sc"]["counts"])], n=n)', "sc = score_dumps(frames, dumps, ends, list(variants), n=n)")
add("TEST boot not fixed", '_require(a.boot == 1000,', '_require(True,')
add("TEST seeds not fixed", '_require(list(a.seeds) == [0, 1],', '_require(True,')
add("TEST b_dump not required", '_require(bool(a.b_dump) and bool(a.control_ends_dump) and bool(a.control_card),', '_require(bool(a.control_ends_dump) and bool(a.control_card),')
add("TEST control_ends not required", '_require(bool(a.b_dump) and bool(a.control_ends_dump) and bool(a.control_card),', '_require(bool(a.b_dump) and bool(a.control_card),')
add("TEST control_card not required", '_require(bool(a.b_dump) and bool(a.control_ends_dump) and bool(a.control_card),', '_require(bool(a.b_dump) and bool(a.control_ends_dump),')
add("TEST scenes not checked", '_require(len({f.scene for f in frames}) == 21,', '_require(True,')
add("sidecar not first (dump_b)", '    write_head(out, dict(split=split, frames=None), prov)                                              # the sidecar FIRST: a crash leaves a sidecar without a pickle, never a pickle without provenance\n', "")
add("sidecar not first (dump_ends)", '    write_head(out, dict(run=run, split=split, frames=None), prov)                                     # the sidecar first, as in dump_b\n', "")
add("pickle overwrite allowed (dump_b)", "    with atomic(out, overwrite=False) as tmp:\n        RO.dump_full(", "    with atomic(out) as tmp:\n        RO.dump_full(")
add("pickle overwrite allowed (dump_ends)", "    save_pickle(out, out_frames, overwrite=False)", "    save_pickle(out, out_frames)")
add("check_seed seed optional", '_require("seed" in ck,', "_require(True,")

anchor = "# glue\n"
add_text = "".join(f" ({n!r}, {o!r}, {w!r}),\n".replace("'" + n + "'", '"' + n + '"', 1) for n, o, w in rows)
assert text.count(anchor) == 1
open(M, "w", encoding="utf-8").write(text.replace(anchor, add_text + anchor))
print(len(rows), "rows added")
