p = "training/grid/jpast.py"
s = open(p, encoding="utf-8").read()


def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:80])
    s = s.replace(a, b)


rep('''it; the state x and w are NOT detached (the gradient runs through the steps); only the input keys of the frames are read (pos, dirv, valid, e): nothing of the GT enters the state. `PastBlock.use` is the kill measure of the card: u = RMS(block(x, past) - block(x, none
key only)) / RMS(x) on the valid tokens. Tests: grid/test_jpast.py (hand-computed values, a loop reference of the block, the invariances, the mutants)''', '''it; the state x and w are NOT detached (the gradient runs through the steps); only the input keys of the frames are read (pos, dirv, valid, e): nothing of the GT enters the state.
`PastBlock.use` is the kill measure of the card: u = RMS(block(x, past) - block(x, past of ANOTHER SAMPLE of the batch)) / RMS(x) on the valid tokens (`roll_past`: every tensor of the past dict rolled by one along the batch axis, the current tokens and e_cur unchanged; a batch of one sample
needs a donor past from outside, `UseMeter.donor`). It is 0 when the block ignores the CONTENT of the past. u_none = RMS(block(x, past) - block(x, none key only)) / RMS(x) is reported next to it and is NOT a test of use: any content-free attention satisfies it, W_o (mean v_past - v_none)
is a constant. Tests: grid/test_jpast.py (hand-computed values, a loop reference of the block, the invariances, a block that ignores the past content, the mutants)''')
rep('''class UseMeter:
    """the kill measure u of the card as a forward hook on a `PastBlock` (the head calls it as block(x, pos, dirv, valid, past)): every call with a non-empty past adds (sum (y - y_none)^2, sum x^2, valid tokens); `take()` -> (u, valid tokens) of the calls since the
    last `take` (None without a call) and clears (`take_sums`: the three sums); `detach()` removes the hook"""

    def __init__(self, block):
        self.acc, self.handle = [0.0, 0.0, 0], block.register_forward_hook(self._hook)

    def _hook(self, block, args, out):
        x, pos, dirv, valid, past = args
        if not past:
            return
        for i, v in enumerate(block.use(x.detach(), pos, dirv, valid, (out - x).detach())):
            self.acc[i] += v

    def take_sums(self):
        s, self.acc = self.acc, [0.0, 0.0, 0]
        return tuple(s)

    def take(self):
        d2, x2, n = self.take_sums()
        return (math.sqrt(d2 / max(x2, 1e-30)), n) if n else None

    def detach(self):
        self.handle.remove()
''', '''def roll_past(past, shift=1):
    """the past of ANOTHER SAMPLE of the batch: every tensor of the past dict rolled by `shift` along the batch axis; e_cur (the existence of the CURRENT tokens, whose queries read the past) and a missing w stay"""
    return {k: (v.roll(shift, 0) if torch.is_tensor(v) and k != "e_cur" else v) for k, v in past.items()}


SUMS = ("d2_none", "x2", "n", "d2_other", "x2_other", "n_other")            # what `PastBlock.use` returns and `UseMeter` adds up


class UseMeter:
    """the kill measures of the card as a forward hook on a `PastBlock` (the head calls it as block(x, pos, dirv, valid, past)): every call with a non-empty past adds the sums of `PastBlock.use`; `donor` = None or a past dict that a batch of ONE sample
    compares with (set by the caller before the forward); `take()` -> dict(u, u_none, n, n_other) of the calls since the last `take` (None without a call; u None without a comparison with another sample's past) and clears (`take_sums`: the sums); `detach()` removes the hook"""

    def __init__(self, block):
        self.acc, self.donor, self.handle = dict.fromkeys(SUMS, 0), None, block.register_forward_hook(self._hook)

    def _hook(self, block, args, out):
        x, pos, dirv, valid, past = args
        if not past:
            return
        for k, v in block.use(x.detach(), pos, dirv, valid, (out - x).detach(), past, self.donor).items():
            self.acc[k] += v

    def take_sums(self):
        s, self.acc = self.acc, dict.fromkeys(SUMS, 0)
        return s

    def take(self):
        s = self.take_sums()
        if not s["n"]:
            return None
        return dict(u=math.sqrt(s["d2_other"] / max(s["x2_other"], 1e-30)) if s["n_other"] else None, u_none=math.sqrt(s["d2_none"] / max(s["x2"], 1e-30)), n=s["n"], n_other=s["n_other"])

    def detach(self):
        self.handle.remove()
''')
rep('''    @torch.no_grad()
    def use(self, x, pos, dirv, valid, y):
        """the kill measure of the card from the block's own update y = block(x, past) - x (as the forward made it): -> (sum of (y - y_none)^2, sum of x^2, valid tokens) over the valid tokens and the D channels, y_none = the update with the none key only;
        u = sqrt(sum1 / sum2) = RMS(y - y_none) / RMS(x), summed over batches before the ratio"""
        m = valid.to(x.dtype)[..., None]
        y0 = self.delta(x, pos, dirv, valid, None)
        return float((((y - y0) * m) ** 2).sum()), float(((x * m) ** 2).sum()), int(valid.sum())
''', '''    @torch.no_grad()
    def use(self, x, pos, dirv, valid, y, past=None, donor=None):
        """the kill measures of the card from the block's own update y = block(x, past) - x (as the forward made it) -> dict of sums over the valid tokens and the D channels (`SUMS`): d2_none = sum (y - y_none)^2 with y_none = the update with the none key only,
        x2 = sum x^2, n = valid tokens; d2_other = sum (y - y_other)^2 with y_other = the update with the past of ANOTHER SAMPLE of the batch (`roll_past(past)`, B >= 2) or with `donor` (a past dict for this batch: a batch of one), x2_other = x2 and n_other = n
        when it was made (0 without: a batch of one sample and no donor). u = sqrt(d2_other / x2_other), u_none = sqrt(d2_none / x2): RMS of the difference / RMS(x), summed over batches before the ratio"""
        m = valid.to(x.dtype)[..., None]
        sq = lambda ref: float((((y - ref) * m) ** 2).sum())  # noqa: E731
        x2, n = float(((x * m) ** 2).sum()), int(valid.sum())
        out = dict(d2_none=sq(self.delta(x, pos, dirv, valid, None)), x2=x2, n=n, d2_other=0.0, x2_other=0.0, n_other=0)
        other = donor if donor is not None else (roll_past(past) if past and x.shape[0] >= 2 else None)
        if other is not None:
            out.update(d2_other=sq(self.delta(x, pos, dirv, valid, other)), x2_other=x2, n_other=n)
        return out
''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
