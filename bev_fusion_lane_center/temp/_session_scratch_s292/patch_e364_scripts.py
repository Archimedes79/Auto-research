import io


def edit(p, pairs):
    s = io.open(p, encoding="utf-8").read()
    for a, b in pairs:
        assert s.count(a) == 1, (p, a[:80], s.count(a))
        s = s.replace(a, b)
    io.open(p, "w", encoding="utf-8").write(s)


# ---- score_nocover.py: COVER_POS -1.0 (<= 0 would still drop an exact coincidence)
edit("temp/e364/score_nocover.py", [
    ("    QE.COVER_POS = 0.0                                                                                     # in process: `cover_pos()` reads the module global at call time\n    assert QE.cover_pos() == 0.0 and QE.cover_pos(None) == 0.0",
     "    QE.COVER_POS = -1.0                                                                                    # in process: `cover_pos()` reads the module global at call time; -1 (not 0) so that even an exact coincidence of ends does not drop a link\n    assert QE.cover_pos() == -1.0 and QE.cover_pos(None) == -1.0"),
    ("with the long-link cover radius `qends_eval.COVER_POS` set to 0.0 IN PROCESS", "with the long-link cover radius `qends_eval.COVER_POS` set to -1.0 IN PROCESS"),
])

# ---- mask_ceiling.py: the true pairs of ALL frames (qends.allowed patched to True while the GT table is built), the model's reality from the stored dumps
edit("temp/e364/mask_ceiling.py", [
    ("import lane_gt_data as LD  # noqa: E402\nimport qendnet as QEN  # noqa: E402\n",
     "import lane_gt_data as LD  # noqa: E402\nimport qendnet as QEN  # noqa: E402\nimport qends as QENDS  # noqa: E402\nimport jwire_data as JD  # noqa: E402\n"),
    ('    LD.QData.DEFAULT_ENDS = True\n    ds, dl = LD.loader(',
     '    # `qends.end_table` builds its candidate list with `allowed` and returns None (an EMPTY table) when no pair of a frame passes it: the true pairs of such frames would be invisible. Patched to True while the table is built (the ends, the clusters and the true pairs do not depend on it).\n    QENDS.allowed = lambda *a, **k: True\n    LD.QData.DEFAULT_ENDS = True\n    ds, dl = LD.loader('),
    ('    tot = dict(pairs=0, admitted=0, fail_turn=0, fail_chord=0, fail_fwd_a=0, fail_fwd_b=0, frames=0)\n',
     '    tot = dict(pairs=0, admitted=0, fail_turn=0, fail_chord=0, fail_fwd_a=0, fail_fwd_b=0, frames=0, frames_with_pairs=0)\n    dumps = {s: {f["row"]: f for f in pickle.load(open(f"{ROOT}/runs/e324_s_s{s}/ends_val.pkl", "rb"))} for s in (0, 1)}\n    real = {s: dict(pairs=0, both_ends_matched=0, reachable=0) for s in (0, 1)}\n'),
    ('            for (u, w) in pr:\n',
     '            tot["frames_with_pairs"] += int(len(pr) > 0)\n            for s in (0, 1):                                                                                  # the model\'s reality: a keypoint matched to each end of the true pair (4 m / 30 degrees, one to one: `jwire_data.match`) and the stored `allowed` True for that keypoint pair\n                e = dumps[s][int(b["row"][i])]\n                k = e["kp_pos"].shape[0] // 2\n                v = np.asarray(e["valid"], bool)\n                m_in = JD.match(e["kp_pos"][:k], e["kp_dir"][:k], v[0], ent, b["ent_m"][i].numpy())\n                m_out = JD.match(e["kp_pos"][k:], e["kp_dir"][k:], v[1], ext, b["ext_m"][i].numpy())\n                al = np.asarray(e["allowed"], bool)\n                for (u, w) in pr:\n                    real[s]["pairs"] += 1\n                    iu, jw = np.where(m_in == u)[0], np.where(m_out == w)[0]\n                    if len(iu) and len(jw):\n                        real[s]["both_ends_matched"] += 1\n                        real[s]["reachable"] += int(any(al[x, y] for x in iu for y in jw))\n            for (u, w) in pr:\n'),
    ('    res = dict(tot, admitted_share=share,',
     '    res = dict(tot, model_reality={str(s): dict(real[s], reachable_of_matched=real[s]["reachable"] / max(real[s]["both_ends_matched"], 1), reachable_of_all=real[s]["reachable"] / max(real[s]["pairs"], 1)) for s in (0, 1)}, admitted_share=share,'),
])
print("ok")
