"""Tests of the cross-attention block to the past keyframe (`jpast.py`, E368).

    wsl bash tools/env/_py14.sh test_jpast.py            # the selftest of jpast.py (hand-computed values, a loop reference of the block, the invariances, the gradient of every parameter, a block that ignores the past content, u and u_none, K = 512 memory), then the hand mutants: every mutant must make it FAIL
    wsl bash tools/env/_py14.sh test_jpast.py selftests  # the selftest only

A mutant = one textual change of the source (a transposed rotation, the translation before the rotation, a flipped sign of the geometry, the none key or a mask or the key weight removed, a detached state, the switch of the key weight inverted,
a GT key read, ...); the mutated copy runs in a temporary folder, four at a time.
"""
import concurrent.futures as cf
import os
import subprocess
import sys
import tempfile

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GRID = os.path.join(ROOT, "training", "grid")
MOD = "jpast.py"
ENV = dict(os.environ, BFLC_ROOT=ROOT, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")      # a shared box: one thread per test process

EIN = 'torch.einsum("bij,bkj->bki", R, '
MUTANTS = [
    ("the warp uses R^T", EIN + "pos) + rel_t", 'torch.einsum("bji,bkj->bki", R, pos) + rel_t'),
    ("the translation is added before the rotation", EIN + "pos) + rel_t.to(pos.dtype)[:, None], torch.einsum(", EIN + "pos + rel_t.to(pos.dtype)[:, None]), torch.einsum("),
    ("the heading is translated too", EIN + "dirv.to(pos.dtype))", EIN + "dirv.to(pos.dtype)) + rel_t.to(pos.dtype)[:, None]"),
    ("the heading is not rotated", EIN + "dirv.to(pos.dtype))", "dirv.to(pos.dtype)"),
    ("the sin of the turn has the other sign", "cross = tq[..., 0] * tk[..., 1] - tq[..., 1] * tk[..., 0]", "cross = tq[..., 1] * tk[..., 0] - tq[..., 0] * tk[..., 1]"),
    ("across is the right normal", "nq = torch.stack([-tq[..., 1], tq[..., 0]], -1)", "nq = torch.stack([tq[..., 1], -tq[..., 0]], -1)"),
    ("the offset is in the world frame", "dot(ch, tq) / 20.0, dot(ch, nq) / 20.0", "ch[..., 1] / 20.0, ch[..., 0] / 20.0"),
    ("the offset is query minus key", "ch = pos_k[:, None, :, :] - pos_q[:, :, None, :]", "ch = pos_q[:, :, None, :] - pos_k[:, None, :, :]"),
    ("e of query and key are swapped", "e_q[:, :, None].expand(ch.shape[:3]), e_k[:, None, :].expand(ch.shape[:3])", "e_k[:, None, :].expand(ch.shape[:3]), e_q[:, :, None].expand(ch.shape[:3])"),
    ("the distance is not scaled", "(dot(ch, ch) + 1e-12).sqrt() / 20.0", "(dot(ch, ch) + 1e-12).sqrt()"),
    ("the cosine is that of the key heading with itself", "dot(tq, tk), cross", "dot(tk, tk), cross"),
    ("before the switch the key weight is the head's own", 'w=out_prev["w"] if learned else e)', 'w=out_prev["w"])'),
    ("after the switch the key weight is still sigmoid(e)", 'w=out_prev["w"] if learned else e)', "w=e)"),
    ("the switch is inverted", 'w=out_prev["w"] if learned else e)', 'w=out_prev["w"] if not learned else e)'),
    ("the state is detached", 'x=out_prev["x"],', 'x=out_prev["x"].detach(),'),
    ("a GT key enters the state", 'valid = _const(frame_prev["valid"]).bool()', 'valid = _const(frame_prev["valid"]).bool() | (frame_prev["ytype"] > 0)'),
    ("e is the logit", 'e = _const(frame_prev["e"]).float().sigmoid()', 'e = _const(frame_prev["e"]).float()'),
    ("the past is not warped", 'pos, dirv = warp_tokens(_const(frame_prev["pos"]).float(), _const(frame_prev["dirv"]).float(), rel_R.float(), rel_t.float())', 'pos, dirv = _const(frame_prev["pos"]).float(), _const(frame_prev["dirv"]).float()'),
    ("e_cur is the previous e", 'e_cur=_const(frame_cur["e"]).float().sigmoid()', "e_cur=e"),
    ("the output projection is not zero-initialised", "nn.init.zeros_(self.o.weight)", "pass"),
    ("the none key is missing", "return torch.cat([z, z_none[..., None]], -1).softmax(-1), v", "return z.softmax(-1), v"),
    ("padding keys are not masked", 'z = z.masked_fill(~past["valid"].bool()[:, None, None, :], NEG)', "z = z"),
    ("padding queries are changed", "return self.o(out.transpose(1, 2).reshape(B, K, D_)) * valid.to(x.dtype)[..., None]", "return self.o(out.transpose(1, 2).reshape(B, K, D_))"),
    ("there is no key weight", 'z = z + (past["w"].float() + 1e-3).log()[:, None, None, :]', "z = z"),
    ("the key weight offset is 0.1", '(past["w"].float() + 1e-3).log()', '(past["w"].float() + 1e-1).log()'),
    ("the attention is not scaled", "(q @ k.transpose(-1, -2)).float() / math.sqrt(self.dh)", "(q @ k.transpose(-1, -2)).float()"),
    ("the geometry bias is missing", "/ math.sqrt(self.dh) + self.bias(g).float().permute(0, 3, 1, 2)", "/ math.sqrt(self.dh)"),
    ("the past state is not normalised", 'self.kv(self.ln_k(past["x"]))', 'self.kv(past["x"])'),
    ("the queries are not normalised", "self.q(self.ln_q(x))", "self.q(x)"),
    ("the past state is detached inside the block", 'self.kv(self.ln_k(past["x"]))', 'self.kv(self.ln_k(past["x"].detach()))'),
    ("there is no residual", "return x + self.delta(x, pos, dirv, valid, past)", "return self.delta(x, pos, dirv, valid, past)"),
    ("the none value is left out of the output", "att[..., :-1].to(v.dtype) @ v + att[..., -1:].to(v.dtype) * vn", "att[..., :-1].to(v.dtype) @ v"),
    ("the query geometry reads the previous e", 'past["e_cur"], past["pos"]', 'past["e"], past["pos"]'),
    ("the none bias is dropped", "/ math.sqrt(self.dh) + self.none_b[None, :, None]", "/ math.sqrt(self.dh)"),
    ("a sample without a previous keyframe keeps its past", "valid = valid & has_prev.bool()[:, None]", "valid = valid"),
    ("the meter counts calls without a past", "        if not past:\n            return\n        for k, v in", "        if False:\n            return\n        for k, v in"),
    ("the meter does not clear", "s, self.acc = self.acc, dict.fromkeys(SUMS, 0)", "s = self.acc"),
    ("the meter ignores the donor", "(out - x).detach(), past, self.donor)", "(out - x).detach(), past)"),
    ("u is taken from the none-key sums", 'math.sqrt(s["d2_other"] / max(s["x2_other"], 1e-30))', 'math.sqrt(s["d2_none"] / max(s["x2"], 1e-30))'),
    ("u_none is taken from the other-sample sums", 'u_none=math.sqrt(s["d2_none"] / max(s["x2"], 1e-30))', 'u_none=math.sqrt(s["d2_other"] / max(s["x2_other"], 1e-30))'),
    ("u: the reference is the none key only", "d2_other=sq(self.delta(x, pos, dirv, valid, other))", "d2_other=sq(self.delta(x, pos, dirv, valid, None))"),
    ("u: the reference is the past itself", "d2_other=sq(self.delta(x, pos, dirv, valid, other))", "d2_other=sq(self.delta(x, pos, dirv, valid, past))"),
    ("u: the roll is missing", "v.roll(shift, 0) if torch.is_tensor(v)", "v if torch.is_tensor(v)"),
    ("u: the roll is by 0", "def roll_past(past, shift=1):", "def roll_past(past, shift=0):"),
    ("u: e_cur is rolled with the past", 'if torch.is_tensor(v) and k != "e_cur" else v', "if torch.is_tensor(v) else v"),
    ("u: a batch of one sample compares with itself", "if past and x.shape[0] >= 2 else None", "if past and x.shape[0] >= 1 else None"),
    ("u: the donor is ignored", "other = donor if donor is not None else (", "other = ("),
    ("u: the other-sample count is not kept", "x2_other=x2, n_other=n", "x2_other=x2, n_other=0"),
    ("u_none: the none-only update is the update with the past", "d2_none=sq(self.delta(x, pos, dirv, valid, None))", "d2_none=sq(self.delta(x, pos, dirv, valid, past))"),
    ("u: padding tokens count in x", "x2, n = float(((x * m) ** 2).sum()), int(valid.sum())", "x2, n = float((x ** 2).sum()), int(valid.sum())"),
    ("u: padding tokens count in the difference", "m = valid.to(x.dtype)[..., None]\n        sq = ", "m = torch.ones_like(x[..., :1])\n        sq = "),
]


def one(args):
    k, (name, a, b), tmp = args
    src = open(os.path.join(GRID, MOD), encoding="utf-8").read()
    if src.count(a) != 1:
        return name, "NOT APPLIED"
    d = os.path.join(tmp, str(k))
    os.makedirs(d)
    p = os.path.join(d, MOD)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(src.replace(a, b))
    r = subprocess.run([sys.executable, p, "selftest"], capture_output=True, text=True, env=ENV)
    return name, "killed" if r.returncode != 0 else "SURVIVED"


def run_mutants(workers=4):
    left = []
    with tempfile.TemporaryDirectory() as tmp, cf.ThreadPoolExecutor(workers) as ex:
        for name, res in ex.map(one, [(k, m, tmp) for k, m in enumerate(MUTANTS)]):
            print(f"{res:<12} {name}", flush=True)
            if res != "killed":
                left.append(name)
    print(f"{len(MUTANTS) - len(left)} of {len(MUTANTS)} killed", flush=True)
    return left


if __name__ == "__main__":
    r = subprocess.run([sys.executable, os.path.join(GRID, MOD), "selftest"], capture_output=True, text=True, env=ENV)
    assert r.returncode == 0, f"{MOD} selftest failed:\n{r.stdout[-800:]}{r.stderr[-1500:]}"
    print(r.stdout.strip().splitlines()[-1], flush=True)
    if len(sys.argv) < 2 or sys.argv[1] != "selftests":
        left = run_mutants()
        assert not left, f"mutants survived: {left}"
    print("test_jpast ok")
