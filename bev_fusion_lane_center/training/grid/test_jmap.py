"""Tests of the cross-attention block to the SD-map tokens (`jmap.py`, E374).

    wsl bash tools/env/_py14.sh test_jmap.py            # the selftest of jmap.py (hand-computed values, a loop reference of the block in float64, the invariances, the gradient of every parameter, a block that ignores the map content, v and v_none, the real head with the hook, K = 512 x M = 704 memory), then the hand mutants: every mutant must make it FAIL
    wsl bash tools/env/_py14.sh test_jmap.py selftests  # the selftest only

A mutant = one textual change of the source (the none key or a mask or a bias removed, a normalisation or the residual dropped, a sign / frame / scale of the relative geometry changed, a field of the embedding dropped, a lane-level key read, ...); the mutated copy runs in a temporary folder, four at a time.
"""
import concurrent.futures as cf
import os
import subprocess
import sys
import tempfile

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GRID = os.path.join(ROOT, "training", "grid")
MOD = "jmap.py"
ENV = dict(os.environ, BFLC_ROOT=ROOT, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")      # a shared box: one thread per test process

MUTANTS = [
    ("the none key is missing", "return torch.cat([z, z_none[..., None]], -1).softmax(-1), v", "return z.softmax(-1), v"),
    ("padding samples are not masked", 'z = z.masked_fill(~maps["valid"].bool()[:, None, None, :], NEG)', "z = z"),
    ("the mask of the samples is the tokens' valid", 'z = z.masked_fill(~maps["valid"].bool()[:, None, None, :], NEG)', "z = z.masked_fill(~valid.bool()[:, None, :, None], NEG)"),
    ("padding queries are changed", "return self.o(out.transpose(1, 2).reshape(B, K, D_)) * valid.to(x.dtype)[..., None]", "return self.o(out.transpose(1, 2).reshape(B, K, D_))"),
    ("the output projection is not zero-initialised", "nn.init.zeros_(self.o.weight)", "pass"),
    ("the output bias is not zero-initialised", "nn.init.zeros_(self.o.bias)", "pass"),
    ("the attention is not scaled", "(q @ k.transpose(-1, -2)).float() / math.sqrt(self.dh)", "(q @ k.transpose(-1, -2)).float()"),
    ("the geometry bias is missing", "/ math.sqrt(self.dh) + self.bias(g).float().permute(0, 3, 1, 2)", "/ math.sqrt(self.dh)"),
    ("the none bias is dropped", "/ math.sqrt(self.dh) + self.none_b[None, :, None]", "/ math.sqrt(self.dh)"),
    ("the map embedding is not normalised", "self.kv(self.ln_k(self.emb(map_features(maps))))", "self.kv(self.emb(map_features(maps)))"),
    ("the queries are not normalised", "self.q(self.ln_q(x))", "self.q(x)"),
    ("there is no residual", "return x + self.delta(x, pos, dirv, valid, maps)", "return self.delta(x, pos, dirv, valid, maps)"),
    ("the none value is left out of the output", "att[..., :-1].to(v.dtype) @ v + att[..., -1:].to(v.dtype) * vn", "att[..., :-1].to(v.dtype) @ v"),
    ("no map is the identity", "return x + self.delta(x, pos, dirv, valid, maps)", "return x + self.delta(x, pos, dirv, valid, maps) if maps else x"),
    ("the use measure is missing", "    use = JP.PastBlock.use ", "    use = None "),
    ("the offset is in the world frame", "dot(ch, tq) / 20.0, dot(ch, nq) / 20.0", "ch[..., 1] / 20.0, ch[..., 0] / 20.0"),
    ("the offset is token minus sample", "ch = mpos[:, None, :, :] - pos[:, :, None, :]", "ch = pos[:, :, None, :] - mpos[:, None, :, :]"),
    ("across is the right normal", "nq = torch.stack([-tq[..., 1], tq[..., 0]], -1)", "nq = torch.stack([tq[..., 1], -tq[..., 0]], -1)"),
    ("the sin of the angle has the other sign", "cross = tq[..., 0] * tm[..., 1] - tq[..., 1] * tm[..., 0]", "cross = tq[..., 1] * tm[..., 0] - tq[..., 0] * tm[..., 1]"),
    ("the cosine is that of the token heading with itself", "dot(tq, tm), cross", "dot(tq, tq), cross"),
    ("the sample's direction is the token's", "tq, tm = dirv[:, :, None, :].expand_as(ch), mdir[:, None, :, :].expand_as(ch)", "tq, tm = dirv[:, :, None, :].expand_as(ch), dirv[:, :, None, :].expand_as(ch)"),
    ("the distance is not scaled", "(dot(ch, ch) + 1e-12).sqrt() / 20.0", "(dot(ch, ch) + 1e-12).sqrt()"),
    ("the count of the geometry is not scaled", "cnt / 4.0, (cnt > 0).to(ch.dtype)], -1)", "cnt, (cnt > 0).to(ch.dtype)], -1)"),
    ("the count-known flag is constant", "(cnt > 0).to(ch.dtype)], -1)", "torch.ones_like(cnt)], -1)"),
    ("the count is missing in the geometry", "cnt / 4.0, (cnt > 0).to(ch.dtype)], -1)", "torch.zeros_like(cnt), (cnt > 0).to(ch.dtype)], -1)"),
    ("the position is not scaled in the embedding", "torch.cat([pos / HALF, JK.fourier(pos),", "torch.cat([pos, JK.fourier(pos),"),
    ("there is no Fourier code in the embedding", "torch.cat([pos / HALF, JK.fourier(pos),", "torch.cat([pos / HALF, torch.zeros_like(JK.fourier(pos)),"),
    ("the direction is not in the embedding", 'maps["dir"].float(), (cnt / 4.0)[..., None]', 'torch.zeros_like(maps["dir"].float()), (cnt / 4.0)[..., None]'),
    ("the count is not in the embedding", "(cnt / 4.0)[..., None], (cnt > 0).float()[..., None]]", "torch.zeros_like(cnt)[..., None], (cnt > 0).float()[..., None]]"),
    ("the count of the embedding is not scaled", "(cnt / 4.0)[..., None], (cnt > 0).float()[..., None]]", "cnt[..., None], (cnt > 0).float()[..., None]]"),
    ("the count-known flag is missing in the embedding", "(cnt > 0).float()[..., None]]", "torch.ones_like(cnt)[..., None]]"),
    ("a lane-level key is read", "(cnt / 4.0)[..., None], (cnt > 0).float()[..., None]]", 'maps.get("frac", cnt / 4.0)[..., None], (cnt > 0).float()[..., None]]'),
    ("the key and the value are the same half", "k, v = self.kv(self.ln_k(self.emb(map_features(maps)))).view(B, M, 2, self.h, self.dh).permute(2, 0, 3, 1, 4)", "k, v = self.kv(self.ln_k(self.emb(map_features(maps)))).view(B, M, 2, self.h, self.dh).permute(2, 0, 3, 1, 4)[[0, 0]]"),
]


def one(args):
    k, (name, a, b), tmp = args
    src = open(os.path.join(GRID, MOD), encoding="utf-8").read()
    if src.count(a) != 1:
        return name, f"NOT APPLIED ({src.count(a)} matches)"
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
    print("test_jmap ok")
