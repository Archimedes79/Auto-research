# ------------------------------------------------------------------------------------------------ B, C
print("B: 'all' - every frame from its own image; C: the gradient comes from the newest frame only")


def model_checks():
    """B and the evaluation-mode half of C on the stub student (fast, a few seconds): name -> bool. Section H runs it under every planted bug"""
    ok = {}
    for F in (2, 3):
        off, on, cur = build(), build(camera_student="stub", student_trainable=True, student_frames="all"), build(camera_student="stub", student_trainable=True)
        fro = build(camera_student="stub", student_init=STUB_CK, student_frames="all")
        b = data(F)
        for label, m in (("trainable", on), ("frozen", fro)):
            want = off.eval().camera_bevs(all_replaced(b, own_outputs(m, b)))
            with torch.no_grad():
                got = m.eval().camera_bevs(b)
            ok[f"F = {F}, {label} student: the BEV equals the student-free model's on the batch whose every frame is the student's output of that frame's own image (oracle one frame at a time)"] = torch.equal(got, want)
        with torch.no_grad():
            got, got_cur, plain = on.camera_bevs(b), cur.eval().camera_bevs(b), off.camera_bevs(b)
            no_old = cur.camera_bevs({k: v for k, v in b.items() if k != "lane_img_old"})
            dim = on.camera_bevs(dict(b, lane_img_old=(b["lane_img_old"].float() * 0.5).to(torch.uint8)))
            junk = on.camera_bevs(dict(b, prob=torch.cat([b["prob"][:, :1], torch.rand_like(b["prob"][:, 1:])], 1), ctx2=torch.cat([b["ctx2"][:, :1], torch.randn_like(b["ctx2"][:, 1:])], 1)))
        ok[f"F = {F}: frame 0 is the same as with 'current'; the older frames differ from the cached teachers' and from 'current'"] = \
            torch.equal(got[:, 0], got_cur[:, 0]) and not torch.allclose(got[:, 1:], plain[:, 1:]) and not torch.allclose(got[:, 1:], got_cur[:, 1:])
        ok[f"F = {F}: 'current' ignores lane_img_old"] = torch.equal(got_cur, no_old)
        ok[f"F = {F}: other older images change the older frames only"] = torch.equal(dim[:, 0], got[:, 0]) and not torch.equal(dim[:, 1:], got[:, 1:])
        ok[f"F = {F}: garbage in the cached older frames changes nothing (they are not read)"] = torch.equal(junk, got)
        ok[f"F = {F}: no lane_img_old: KeyError that names it and the builder"] = raises(KeyError, lambda: on.camera_bevs({k: v for k, v in b.items() if k != "lane_img_old"}), "lane_img_old", "build-cache-older")
        ok[f"F = {F}: one image too many: ValueError"] = raises(ValueError, lambda: on.camera_bevs(dict(b, lane_img_old=torch.cat([b["lane_img_old"], b["lane_img_old"][:, :1]], 1))), "older frame")
    on, cur = build(camera_student="stub", student_trainable=True, student_frames="all"), build(camera_student="stub", student_trainable=True)
    b1 = data(1)
    with torch.no_grad():
        ok["'all' with a one-frame batch and no lane_img_old = 'current' (the stream step needs no older image)"] = torch.equal(on.eval().camera_bevs(b1), cur.eval().camera_bevs(b1))
    b3 = data(3)
    side = on(b3, dts=(0.0,))["student_side"]
    want = on.student(b3["lane_img"])
    ok["student_side (the anchor) is frame 0's: the student's outputs of lane_img against the cached teacher's frame 0"] = \
        all(torch.equal(side[k], want[n + k[1:]]) for k, n in (("p16", "prob"), ("c16", "ctx"), ("p8", "prob"), ("c8", "ctx"))) and torch.equal(side["t_p16"], b3["prob"][:, 0]) and torch.equal(side["t_c8"], b3["ctx2"][:, 0])
    # C, evaluation mode (BatchNorm fixed): exactly the newest frame
    sc = on.student.scale
    w = torch.randn(on.camera_bevs(b3).shape, generator=torch.Generator().manual_seed(5))
    g_cur = torch.autograd.grad(on.camera_bevs(b3)[:, 0].sum(), sc, allow_unused=True)[0]
    g_old = torch.autograd.grad(on.camera_bevs(b3)[:, 1:].sum(), sc, allow_unused=True)[0]
    ok["C evaluation mode: the newest frame's BEV reaches the student, the older frames' BEV does not (no graph)"] = g_cur is not None and float(g_cur.abs()) > 0 and (g_old is None or float(g_old.abs()) == 0.0)
    g_a = torch.autograd.grad((on.camera_bevs(b3) * w).sum(), sc)[0]
    g_b = torch.autograd.grad((on.camera_bevs(dict(b3, lane_img_old=older_images(2, 2).flip(1))) * w).sum(), sc)[0]
    ok["C evaluation mode: the student's gradient is unchanged when the older frames' images are replaced"] = torch.equal(g_a, g_b) and float(g_a.abs()) > 0
    old_out = on._run_student_older(b3["lane_img_old"])
    ok["C the older frames' student outputs carry no graph (gradients on), are (B, F - 1, ...) and only prob / ctx of the two heads"] = \
        all(not v.requires_grad for v in old_out.values()) and all(v.shape[:2] == (2, 2) for v in old_out.values()) and sorted(old_out) == ["ctx16", "ctx8", "prob16", "prob8"]
    cur_out = on._run_student(on._student_images(b3, 3)[0])
    ok["C the current frame's outputs carry the graph of a trainable student"] = all(v.requires_grad for k, v in cur_out.items() if k.startswith(("prob", "ctx", "logit")))
    return ok


for name, v in model_checks().items():
    check(v, name)
# the real encoder: the batch of older frames equals the frames made one at a time (flatten / view with real convolutions, BatchNorm in eval mode)
off_r, on_r = build(), build(camera_student="resnet18", student_trainable=True, student_frames="all")
b3 = data(3)
with torch.no_grad():
    want_r = off_r.eval().camera_bevs(all_replaced(b3, own_outputs(on_r, b3)))
    got_r = on_r.eval().camera_bevs(b3)
    swapped = on_r.camera_bevs(dict(b3, lane_img_old=b3["lane_img_old"].flip(0)))
check(torch.allclose(got_r, want_r, atol=1e-5, rtol=1e-4) and not torch.allclose(swapped[:, 1:], got_r[:, 1:]), f"real ResNet-18 student: the batch of older frames equals the frames made one at a time (max diff {float((got_r - want_r).abs().max()):.1e}); swapping the samples' images changes the older frames")
# C, training mode, the real student (checkpoint_encode, the lift recomputed in the backward): the older frames are constants - output, gradients and state equal the oracle's
for K in (1, 2):
    tr = build(camera_student="resnet18", student_trainable=True, student_frames="all")
    st = tr.student
    bt = data(K + 1)
    state_all = copy.deepcopy(tr.state_dict())
    w_all = torch.randn_like(tr.eval().camera_bevs(bt))
    runs = {}
    for oracle in (False, True):
        tr.load_state_dict(state_all)
        tr.train()
        tr.zero_grad(set_to_none=True)
        st.checkpoint_encode = not oracle
        out_ = as_head(tr).camera_bevs(all_replaced(bt, own_outputs(tr, bt))) if oracle else tr.camera_bevs(bt)       # the oracle: the code before, cached outputs = the student's, older frames constants
        (out_ * w_all).sum().backward()
        runs[oracle] = (out_.detach().clone(), {n: p.grad.clone() for n, p in tr.named_parameters() if p.grad is not None}, {k: v.clone() for k, v in tr.state_dict().items()})
    live, ref = runs[False], runs[True]
    if K == 1:
        ok_ = same_snap(live, ref)
    else:                                                                                 # one batch of 2 B frames against frames one at a time: real convolutions differ in the last bits
        ok_ = torch.allclose(live[0], ref[0], atol=1e-6, rtol=1e-5) and set(live[1]) == set(ref[1]) and all(torch.allclose(live[1][n], ref[1][n], atol=1e-5, rtol=1e-4) for n in ref[1]) \
            and all(torch.allclose(live[2][k].float(), ref[2][k].float(), atol=1e-6, rtol=1e-5) for k in ref[2])
    check(ok_, f"C training mode, F = {K + 1}, real student: output, all {len(ref[1])} gradients and all {len(ref[2])} state tensors equal the oracle whose older frames are constants ({'bit for bit' if K == 1 else 'to 1e-5'})")
