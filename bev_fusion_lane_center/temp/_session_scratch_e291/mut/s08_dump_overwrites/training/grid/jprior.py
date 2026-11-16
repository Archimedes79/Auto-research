"""E291: the number of lanes per direction of a road-level map as a SOFT, GATED input of the junction pair head (`jends.JEnds(prior=True)`).

    table(jg, token, spec, partner=None) -> dict(c (A, 2), t (A, 2), n_in (A,), n_out (A,))   the arm table of a frame: the arms of `jgt.junction_gt(g)["arms"]` with ONLY the centre, the heading into the junction and the numbers of
                                                       entry / exit GT ends (the lane count per direction of a map), corrupted by a PURE function of (frame token, zone, rate); `jg is None` -> an empty table. The zone
                                                       decides the rate and the junction cases inside `table`; it is not carried by the table
    collate(tables) -> the same keys as padded tensors (B, A, .); a pad row has n_in = n_out = PAD, an UNKNOWN count is UNK (-1), an arm the map does not have is absent (not in the table)
    features(pos, D, kind, ok, arms) -> (B, N, NF) per keypoint: [count / 4, unknown or absent flag, distance to the arm axis / 20 m, heading agreement, lateral rank / max(count, 1), (real keypoints found - count) / 4], and the arm index
    PriorHead(d)(pos, D, valid, real, arms) -> (term (B, 2K, d), info): term = sigmoid(g) * f, zero at the start (the head starts as the one without it)

What a keypoint sees: the arms it is eligible for are those within RADIUS (10 m) of their centre whose travel direction (the heading into the junction for an entry, the opposite for an exit) is within 60 degrees of the keypoint's heading; its own arm is
the one of them nearest to its axis, the count is the one of its OWN direction. The keypoints of an arm are all the real ones eligible for it (the quantity of the step 0 of the card). Only REAL keypoints (peaks of the heat: `valid & real`, never the GT end cells that `JEnds.forward` injects in training) are counted and ranked, so the features of a keypoint are the same in training
and at evaluation. The positions are DETACHED: the gradient of the prior path stops at the keypoints (the map cannot shape the heat in E291; named hard steps: the nearest-arm argmin, the presence radius, the rank sort).
The read-outs (`Spec.readout`, names `clean`, `noise10`, `noise30`, `pm1`, `u`, `s`) and the two training specs (`Spec.train`) are in one place; S = the table of another frame: the first frame with a junction GT at or after position (index + N // 2) of the split (`partners`).
Run the self-test: python jprior.py
"""
import dataclasses
import hashlib

import numpy as np
import torch
import torch.nn as nn

PAD, UNK = -2, -1                  # n_in / n_out of a pad row, of an UNKNOWN count
KEYS = ("c", "t", "n_in", "n_out")   # the table: nothing else (no spread, no zone, no connection, no lane id)
RADIUS, COS_MIN = 10.0, 0.5        # a keypoint belongs to an arm within RADIUS m of its centre (the centre is the midpoint of the entry and exit centroids, up to 14 m apart: 1 % of the ends lie farther than 10 m) and 60 degrees of its direction
NF = 6
MAX_WRONG, SHIFT_M, YAW_DEG = 5, 2.0, 5.0     # a wrong count is drawn from {0 .. 5}; an arm moves by up to 2 m and turns by up to 5 degrees
RATES = (0.0, 0.1, 0.3)            # the rate of a junction in training with `mix`
PM1_RATE = 0.3                     # the +-1 read-out: every count moves by one with this probability
MODES = ("clean", "noise", "pm1", "U", "S", "mix")


@dataclasses.dataclass(frozen=True)
class Spec:
    """what is done to the table of a frame. mode: clean | noise (wrong counts with probability `rate` per arm and direction, an arm missing with probability rate / 3, the arms moved and turned, only when rate > 0) | pm1 (every count +-1 with
    probability `rate`) | U (every count unknown, geometry kept) | S (the table of ANOTHER frame) | mix (training: the rate of each junction is drawn from RATES). unknown: the share of arms whose counts are unknown (both directions).
    drop, ujunc: shares of the junctions without a map / with all counts unknown and the geometry kept (training only: the read-outs have none)."""
    mode: str = "clean"
    rate: float = 0.0
    unknown: float = 0.0
    drop: float = 0.0
    ujunc: float = 0.0

    def __post_init__(self):
        assert self.mode in MODES, self.mode

    @staticmethod
    def train(noise):
        """the training spec of `--prior-noise {0, mix}`: 20 % of the junctions have no map; mix: besides, 20 % of the junctions are 'counts unknown, geometry kept', the rate of each other junction from RATES, 15 % of its arms unknown"""
        assert noise in ("0", "mix"), noise
        return Spec("clean", drop=0.2) if noise == "0" else Spec("mix", unknown=0.15, drop=0.2, ujunc=0.2)

    @staticmethod
    def readout(mode, rate=0.0):
        return Spec(mode, rate=PM1_RATE if mode == "pm1" else (rate if mode == "noise" else 0.0))

    @staticmethod
    def parse(name):
        """the read-out of a name: clean, noise<percent>, pm1, u, s"""
        if name.startswith("noise"):
            return Spec.readout("noise", int(name[5:]) / 100.0)
        assert name in ("clean", "pm1", "u", "s"), f"read-out {name!r}: clean, noise<percent>, pm1, u or s"
        return Spec.readout({"u": "U", "s": "S"}.get(name, name))

    @property
    def name(self):
        """the read-out's name in a run folder and a card tag (lower case letters and digits)"""
        return f"noise{round(100 * self.rate)}" if self.mode == "noise" else self.mode.lower()


def readouts(modes, rate=0.0):
    """the specs of `jends_eval.py dump --prior-mode`: a comma list of clean, noise10 .., pm1, u, s (case free), or `noise` with the rate of --prior-noise; one pass over the loader writes them all"""
    out = [Spec.readout("noise", rate) if w.strip().lower() == "noise" else Spec.parse(w.strip().lower()) for w in modes.split(",")]
    assert all(s.rate > 0 for s in out if s.mode == "noise"), "--prior-mode noise needs --prior-noise r > 0"
    assert len({s.name for s in out}) == len(out), f"a read-out twice in {modes!r}"
    return out


def _h(*key):
    """a stable 64 bit integer of a key: the same on every machine and in every worker (python's hash() is salted per process)"""
    return int.from_bytes(hashlib.sha256("|".join(str(k) for k in key).encode()).digest()[:8], "little")


def _u(*key):
    return (_h(*key) >> 11) / float(1 << 53)


def _wrong(n, u):
    """a count in {0 .. MAX_WRONG} different from n, chosen by u in [0, 1)"""
    cand = [v for v in range(MAX_WRONG + 1) if v != n]
    return cand[min(int(u * len(cand)), len(cand) - 1)]


def _zone_rate(spec, token, zone):
    return RATES[min(int(_u("rate", token, zone) * len(RATES)), len(RATES) - 1)] if spec.mode == "mix" else (spec.rate if spec.mode in ("noise", "pm1") else 0.0)


def _corrupt_zone(arms, token, zone, spec):
    """the arms (c, t, n_in, n_out) of one junction after the corruption: a function of (token, zone, rate) and the arms. Every arm draws the same nine numbers in the same order whatever the mode, so a rule never shifts the draws of another."""
    r = _zone_rate(spec, token, zone)
    rs = np.random.RandomState(_h("arms", token, zone) % (1 << 32))
    step = lambda n, ev, sg: n + (1 if sg < 0.5 or n == 0 else -1) if ev < r else n  # noqa: E731
    out = []
    for c, t, ni, no in arms:
        u = rs.random_sample(9)                                  # 0 arm missing, 1 unknown, 2 shift length, 3 shift direction, 4 yaw, 5 / 7 count in / out wrong, 6 / 8 its value (pm1: its sign)
        c, t, ni, no = np.asarray(c, np.float64), np.asarray(t, np.float64), int(ni), int(no)
        if spec.mode in ("noise", "mix"):
            if u[0] < r / 3.0:
                continue                                         # a road the map does not have
            if r > 0:                                            # a junction with no error at all (r = 0) is the clean table
                a, y = 2 * np.pi * u[3], np.radians(YAW_DEG) * (2 * u[4] - 1)
                c = c + SHIFT_M * u[2] * np.array([np.cos(a), np.sin(a)])
                t = np.array([t[0] * np.cos(y) - t[1] * np.sin(y), t[0] * np.sin(y) + t[1] * np.cos(y)])
            ni, no = (_wrong(ni, u[6]) if u[5] < r else ni), (_wrong(no, u[8]) if u[7] < r else no)
            if u[1] < spec.unknown:
                ni = no = UNK
        elif spec.mode == "pm1":
            ni, no = step(ni, u[5], u[6]), step(no, u[7], u[8])
        elif spec.mode == "U":
            ni = no = UNK
        out.append((c, t, ni, no))
    return out


def table(jg, token, spec, partner=None):
    """the arm table of a frame (numpy). `jg` = `jgt.junction_gt(g)` or None; `partner(token)` -> the `junction_gt` of another frame (mode S: `partners`). Per junction (zone) ONE draw u of (token, zone) decides: u < drop: no map, then
    u < drop + ujunc: all counts unknown and the geometry kept, else the corruption of the spec (arms with zone -1 are one junction)."""
    if spec.mode == "S":
        jg = partner(token)
    zones = {}
    for a in ([] if jg is None else jg["arms"]):
        zones.setdefault(int(a["zone"]), []).append((a["c"], a["t"], a["n_in"], a["n_out"]))
    rows = []
    for z in sorted(zones):
        u = _u("junction", token, z)
        if u < spec.drop:
            continue                                             # a junction the map does not cover
        rows += _corrupt_zone(zones[z], token, z, Spec("U") if u < spec.drop + spec.ujunc else spec)
    return dict(c=np.array([r[0] for r in rows], np.float32).reshape(-1, 2), t=np.array([r[1] for r in rows], np.float32).reshape(-1, 2),
                n_in=np.array([r[2] for r in rows], np.int64), n_out=np.array([r[3] for r in rows], np.int64))


def partners(token, tokens):
    """the other frames in the order of the S read-out: the frame at position (index of token + N // 2) of the split, then on; a frame is never its own partner (the caller takes the first one with a junction GT)"""
    n = len(tokens)
    s = tokens.index(token) + n // 2
    for i in range(n):
        if tokens[(s + i) % n] != token:
            yield tokens[(s + i) % n]


def collate(tables):
    B, A = len(tables), max(1, max(len(x["n_in"]) for x in tables))
    out = dict(c=torch.zeros(B, A, 2), t=torch.zeros(B, A, 2), n_in=torch.full((B, A), PAD, dtype=torch.long), n_out=torch.full((B, A), PAD, dtype=torch.long))
    for i, x in enumerate(tables):
        assert tuple(x) == KEYS, f"the table has the keys {tuple(x)}, not {KEYS}"
        n = len(x["n_in"])
        out["c"][i, :n], out["t"][i, :n] = torch.from_numpy(x["c"]), torch.from_numpy(x["t"])
        out["n_in"][i, :n], out["n_out"][i, :n] = torch.from_numpy(x["n_in"]), torch.from_numpy(x["n_out"])
    return out


@torch.no_grad()
def features(pos, D, kind, ok, arms):
    """pos, D (B, N, 2): keypoint position (m) and unit heading; kind (B, N): 0 entry, 1 exit; ok (B, N) bool: a REAL peak (valid & real); arms: the collated table.
    -> f (B, N, NF), arm (B, N) long (index of the arm, -1 none). Absent (no arm in reach) = all zero with the flag 1; unknown count = flag 1, count / rank / found terms 0, the geometry kept."""
    B, N = pos.shape[:2]
    ex = (kind == 1)[:, :, None]
    n = torch.where(ex, arms["n_out"][:, None, :], arms["n_in"][:, None, :])                          # (B, N, A) the count of the keypoint's own direction
    ref = torch.where(ex[..., None], -arms["t"][:, None], arms["t"][:, None])                          # (B, N, A, 2) the travel direction of the kind
    rel = pos[:, :, None, :] - arms["c"][:, None]
    lat = (rel * torch.stack([-ref[..., 1], ref[..., 0]], -1)).sum(-1)                                  # signed offset to the left of the travel direction (rank 0 = leftmost, as `jgt._ranks`)
    cos = (D[:, :, None, :] * ref).sum(-1)
    elig = (arms["n_in"] > PAD)[:, None, :] & (rel.norm(dim=-1) <= RADIUS) & (cos >= COS_MIN)
    arm = torch.where(elig, lat.abs(), torch.full_like(lat, float("inf"))).argmin(-1)                 # the nearest arm axis
    has = elig.any(-1)
    pick = lambda x: x.gather(2, arm[..., None])[..., 0]  # noqa: E731
    n_i, lat_i, cos_i = pick(n).float(), pick(lat), pick(cos)
    own = arm[:, None, :].expand(B, N, N)                                                              # [b, k, i] = the arm of keypoint i
    mem = elig.gather(2, own) & ok[:, :, None] & (kind[:, :, None] == kind[:, None, :])               # [b, k, i]: k is a REAL keypoint of the arm of i, in the direction of i (an arm's keypoints are the ones eligible for it: a keypoint near two arms counts for both)
    lat_k = lat.gather(2, own)                                                                          # the lateral offset of k to the axis of the arm of i
    idx = torch.arange(N, device=pos.device)
    left = mem & ((lat_k > lat_i[:, None, :]) | ((lat_k == lat_i[:, None, :]) & (idx[None, :, None] < idx[None, None, :])))
    rank, found = left.sum(1).float(), mem.sum(1).float()
    known = has & (pick(n) != UNK)
    z = torch.zeros_like(n_i)
    f = torch.stack([torch.where(known, n_i / 4.0, z), (~known).float(), torch.where(has, lat_i.abs() / 20.0, z), torch.where(has, cos_i, z),
                     torch.where(known, rank / n_i.clamp_min(1.0), z), torch.where(known, (found - n_i) / 4.0, z)], -1)
    return f, torch.where(has, arm, torch.full_like(arm, -1))


class PriorHead(nn.Module):
    """token += sigmoid(g) * f: `gate` (6 -> 16 -> 1) the gate logit, `feat` (6 -> d) the feature, both from the six keypoint features. The output layer of the gate and `feat` start at zero: the term is exactly zero and the head
    starts as the one without it; the gradient reaches `feat` at the first step and the gate from the second (the gate's gradient is proportional to f)."""

    def __init__(self, d=64, hid=16):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(NF, hid), nn.GELU(), nn.Linear(hid, 1))
        self.feat = nn.Linear(NF, d)
        for p in (self.gate[2].weight, self.gate[2].bias, self.feat.weight, self.feat.bias):
            nn.init.zeros_(p)

    def forward(self, pos, D, valid, real, arms):
        """pos, D (B, 2K, 2) (detached here), valid / real (B, 2, K) bool as `JEnds.forward`, arms the collated table -> (term (B, 2K, d), info of detached (B, 2K) tensors: gate, term (norm), n (the count shown, -1 = unknown or absent), arm)"""
        B, K = valid.shape[0], valid.shape[2]
        kind = torch.cat([torch.zeros(B, K), torch.ones(B, K)], 1).to(pos.device)
        v2 = valid.flatten(1)
        f, arm = features(pos.detach().float(), D.detach().float(), kind, (valid & real).flatten(1), arms)
        g = torch.sigmoid(self.gate(f))
        term = g * self.feat(f) * v2[..., None]
        shown = torch.where(f[..., 1] > 0, torch.full_like(f[..., 0], -1.0), f[..., 0] * 4.0)
        return term, dict(gate=g[..., 0].detach().float(), term=term.detach().float().norm(dim=-1), n=shown, arm=arm)


def _toy(n_in=3, n_out=2):
    """one arm at (0, -10) heading north into the junction, entry keypoints on three lanes at x = -3.5, 0, 3.5, an exit keypoint at (-1.75, -10) heading south, one keypoint far away"""
    arms = collate([dict(c=np.array([[0.0, -10.0]], np.float32), t=np.array([[0.0, 1.0]], np.float32), n_in=np.array([n_in]), n_out=np.array([n_out]))])
    pos = torch.tensor([[[-3.5, -10.0], [0.0, -10.0], [3.5, -10.0], [-1.75, -10.0], [30.0, 30.0]]])
    D = torch.tensor([[[0.0, 1.0]] * 3 + [[0.0, -1.0], [0.0, 1.0]]])
    kind = torch.tensor([[0, 0, 0, 1, 0]])
    return arms, pos, D, kind


def _selftest():
    fake = lambda zone, n=(2, 3): dict(c=np.array([0.0, -10.0]), t=np.array([0.0, 1.0]), n_in=n[0], n_out=n[1], zone=zone)  # noqa: E731
    same_table = lambda a, b: all(a[k].shape == b[k].shape and np.array_equal(a[k], b[k]) for k in KEYS)  # noqa: E731
    jg = dict(arms=[fake(0), fake(0, (1, 1)), fake(1), fake(1, (4, 5))])
    assert RADIUS == 10.0, "the association radius of the card (v4)"
    # leak: the table has the four keys and nothing of the zone / spread / ids
    spec = Spec.readout("noise", 0.3)
    tb = table(jg, "tok", spec)
    assert tuple(tb) == KEYS and tuple(collate([tb])) == KEYS
    # a pure function: the same (token, zone, rate) gives the same table in any process (the hash is pinned) and any order of the calls; another token another table; no junction GT: an empty table
    assert _h("tok", 0) == 7209337710685028205 and same_table(tb, table(jg, "tok", spec))
    many = dict(arms=[dict(fake(k)) for k in range(4000)])                                           # 4000 junctions of one arm each: the statistics
    assert not same_table(table(many, "tok", spec), table(many, "tok2", spec))
    kept = np.mean([len(table(dict(arms=[fake(0)]), f"t{i}", Spec.train("0"))["n_in"]) for i in range(400)])
    assert 0.7 < kept < 0.9, f"the junction draw is a function of (token, zone): a fifth of the frames lose the map, {kept}"
    assert {len(table(dict(arms=[fake(-1)] * 3), f"t{i}", Spec.train("0"))["n_in"]) for i in range(200)} == {0, 3}, "the arms of zone -1 are ONE junction: all in the table or none"
    assert len(table(None, "tok", spec)["n_in"]) == 0 and collate([table(None, "tok", spec)])["n_in"].tolist() == [[PAD]]
    # the wrong-count rate: with rate r a count is wrong in about r of the arms and directions and never equal to the truth; an arm is missing with probability r / 3; no unknown arm; shift <= 2 m, turn <= 5 degrees
    for r in (0.1, 0.3):
        got = table(many, "t", Spec.readout("noise", r))
        wrong = int((got["n_in"] != 2).sum() + (got["n_out"] != 3).sum())
        assert abs(wrong / (2 * len(got["n_in"])) - r) < 0.025, (r, wrong / (2 * len(got["n_in"])))
        assert abs(len(got["n_in"]) / 4000 - (1 - r / 3)) < 0.02, (r, len(got["n_in"]))
        assert ((got["n_in"] >= 0) & (got["n_in"] <= MAX_WRONG)).all() and ((got["n_out"] >= 0) & (got["n_out"] <= MAX_WRONG)).all()
        assert np.abs(got["c"] - np.array([0.0, -10.0])).max() <= SHIFT_M + 1e-5 and np.abs(np.degrees(np.arctan2(-got["t"][:, 0], got["t"][:, 1]))).max() <= YAW_DEG + 1e-3 and np.abs(got["c"][:, 1] + 10.0).max() > 1.0
    cl = table(many, "t", Spec("clean"))
    assert (cl["n_in"] == 2).all() and (cl["n_out"] == 3).all() and np.allclose(cl["c"], [0.0, -10.0]) and len(cl["n_in"]) == 4000
    u = table(many, "t", Spec.readout("U"))
    assert (u["n_in"] == UNK).all() and (u["n_out"] == UNK).all() and np.allclose(u["c"], [0.0, -10.0]) and np.allclose(u["t"], [0.0, 1.0])
    pm = table(many, "t", Spec.readout("pm1"))                                                          # every count moves by one with probability 0.3, geometry kept, no arm missing
    for got, n0 in ((pm["n_in"], 2), (pm["n_out"], 3)):
        assert abs(float((got != n0).mean()) - PM1_RATE) < 0.03 and set(np.unique(np.abs(got - n0))) == {0, 1}
    assert np.allclose(pm["c"], [0.0, -10.0]) and len(pm["n_in"]) == 4000
    z0 = table(dict(arms=[dict(fake(k), n_in=0, n_out=0) for k in range(2000)]), "t", Spec.readout("pm1"))
    assert z0["n_in"].min() == 0 and z0["n_in"].max() == 1, "a count of 0 can only go up"
    assert Spec.parse("noise30") == Spec.readout("noise", 0.3) and Spec.parse("u").mode == "U" and Spec.parse("s").mode == "S" and [Spec.parse(n).name for n in ("clean", "noise10", "pm1", "u", "s")] == ["clean", "noise10", "pm1", "u", "s"]
    assert [s.name for s in readouts("clean,Noise,U,s,pm1", 0.1)] == ["clean", "noise10", "u", "s", "pm1"]
    for bad in ("noise", "clean,clean", "noise7x"):
        try:
            readouts(bad)
            raise RuntimeError(bad)
        except (AssertionError, ValueError):
            pass
    # the rate of a junction in training: a third each at 0 / 10 / 30 %; a junction with r = 0 is the clean table (no shift, no turn, no wrong count: the unknown arms apart); r computed from (token, zone)
    mix = Spec.train("mix")
    tr = table(many, "t", mix)
    r_of = [RATES[min(int(_u("rate", "t", k) * 3), 2)] for k in range(4000)]
    assert [round(r_of.count(x) / 4000, 1) for x in RATES] == [0.3, 0.3, 0.3] and abs(r_of.count(0.1) / 4000 - 1 / 3) < 0.03
    rows = {k: _corrupt_zone([(a["c"], a["t"], a["n_in"], a["n_out"])], "t", k, mix) for k, a in enumerate(many["arms"])}
    for k in range(400):
        if r_of[k] == 0.0:
            for c, t, ni, no in rows[k]:
                assert np.allclose(c, [0.0, -10.0]) and np.allclose(t, [0.0, 1.0]) and (ni, no) in ((2, 3), (UNK, UNK)), (ni, no)
    # ... a fifth of the junctions without a map, a fifth with all counts unknown and the geometry kept, the rest the rate's corruption with 15 % unknown arms (the shares follow from the draws)
    rest = 1 - 0.4
    kept = 0.2 + rest * (1 - np.mean([x / 3 for x in RATES]))
    assert abs(len(tr["n_in"]) / 4000 - kept) < 0.03, (len(tr["n_in"]) / 4000, kept)
    assert abs(float((tr["n_in"] == UNK).mean()) - (0.2 + rest * (1 - np.mean([x / 3 for x in RATES])) * 0.15) / kept) < 0.03
    a0 = table(many, "t", Spec.train("0"))
    assert abs(len(a0["n_in"]) / 4000 - 0.8) < 0.03 and (a0["n_in"] == 2).all() and (a0["n_out"] == 3).all()
    s = table(jg, "tok", Spec.readout("S"), partner=lambda tk: dict(arms=[fake(0, (4, 4))]))          # S: the table of another frame, deterministic
    assert s["n_in"].tolist() == [4] and table(jg, "tok", Spec.readout("S"), partner=lambda tk: None)["n_in"].shape == (0,)
    order = list(partners("b", ["a", "b", "c", "d", "e"]))
    assert order == ["d", "e", "a", "c"] and sorted(partners("a", ["a", "b"])) == ["b"] and all(list(partners(t, list("abcdef")))[0] == list("abcdef")[(i + 3) % 6] for i, t in enumerate("abcdef"))
    # the features, known answer: an arm with 3 entry and 2 exit lanes, three real entry keypoints on its three lanes, one real exit keypoint, one keypoint far away
    arms, pos, D, kind = _toy()
    ok = torch.ones(1, 5, dtype=torch.bool)
    f, arm = features(pos, D, kind, ok, arms)
    assert arm.tolist() == [[0, 0, 0, 0, -1]]
    want = torch.tensor([[0.75, 0, 3.5 / 20, 1, 0, 0], [0.75, 0, 0, 1, 1 / 3, 0], [0.75, 0, 3.5 / 20, 1, 2 / 3, 0],      # rank 0 = leftmost: x = -3.5 is on the LEFT of a north heading (left normal (-1, 0))
                         [0.5, 0, 1.75 / 20, 1, 0, (1 - 2) / 4], [0, 1, 0, 0, 0, 0]])                              # the exit: its own count (2) and its own found (1); the far keypoint is absent: flag only
    assert torch.allclose(f[0], want, atol=1e-6), f[0]
    assert torch.allclose(features(pos, D, kind, torch.tensor([[0, 1, 1, 1, 0]], dtype=torch.bool), arms)[0][0, :3, 5], torch.tensor([-1 / 4, -1 / 4, -1 / 4])), "an injected (not real) keypoint is not counted"
    far = lambda dy: features(torch.tensor([[[0.0, -10.0 + dy]]]), torch.tensor([[[0.0, 1.0]]]), torch.tensor([[0]]), torch.ones(1, 1, dtype=torch.bool), arms)[1].item()  # noqa: E731
    assert far(-9.0) == 0 and far(9.0) == 0 and far(-11.0) == -1 and far(11.0) == -1, "the presence radius is 10 m"
    two = collate([dict(c=np.array([[0.0, -10.0], [1.5, -4.0]], np.float32), t=np.array([[0.0, 1.0]] * 2, np.float32), n_in=np.array([1, 2]), n_out=np.array([0, 0]))])    # two arms 6 m apart along one road: a keypoint belongs to the arm of the nearer axis
    f2, a2 = features(torch.tensor([[[-1.0, -10.0], [1.0, -4.0]]]), torch.tensor([[[0.0, 1.0]] * 2]), torch.tensor([[0, 0]]), torch.ones(1, 2, dtype=torch.bool), two)               # and counts for EVERY arm it is eligible for
    assert a2.tolist() == [[0, 1]] and torch.allclose(f2[0, :, 5], torch.tensor([(2 - 1) / 4, (2 - 2) / 4])) and torch.allclose(f2[0, :, 4], torch.tensor([0.0, 0.5])), f2
    fu, _ = features(pos, D, kind, ok, collate([dict(c=np.array([[0.0, -10.0]], np.float32), t=np.array([[0.0, 1.0]], np.float32), n_in=np.array([UNK]), n_out=np.array([UNK]))]))
    assert torch.allclose(fu[0, :4, 1], torch.ones(4)) and torch.allclose(fu[0, :4, 2], want[:4, 2]) and torch.allclose(fu[0, :4, 3], want[:4, 3]) and float(fu[0, :4][:, [0, 4, 5]].abs().sum()) == 0.0, fu   # unknown: flagged, the geometry stays
    f0, _ = features(pos, D, kind, ok, _toy(0, 0)[0])
    assert torch.allclose(f0[0, :3, 0], torch.zeros(3)) and float(f0[0, 0, 1]) == 0.0 and torch.allclose(f0[0, :3, 4], torch.tensor([0.0, 1.0, 2.0])) and torch.allclose(f0[0, :3, 5], torch.full((3,), 3 / 4)), "a known count of 0 is no unknown: rank / max(n, 1), found / 4"
    # an exit keypoint heading INTO the junction does not belong to the arm (the direction of its kind)
    fe, ae = features(pos[:, 3:4], torch.tensor([[[0.0, 1.0]]]), torch.tensor([[1]]), ok[:, :1], arms)
    assert ae.tolist() == [[-1]] and float(fe[0, 0, 1]) == 1.0
    # the head: zero term at the start, a gradient into the feature layer at the first step and into the gate from the next, nothing into the keypoint positions, every parameter moves
    head = PriorHead(d=8)
    p6, D6 = torch.cat([pos[:, :5], pos[:, :1]], 1), torch.cat([D[:, :5], D[:, :1]], 1)
    valid = real = torch.ones(1, 2, 3, dtype=torch.bool)
    term, info = head(p6, D6, valid, real, arms)
    assert term.shape == (1, 6, 8) and float(term.detach().abs().max()) == 0.0 and info["n"].shape == (1, 6)
    before = [p.detach().clone() for p in head.parameters()]
    opt = torch.optim.SGD(head.parameters(), lr=0.1)
    for step in range(3):
        pt = p6.clone().requires_grad_(True)
        term, info = head(pt, D6, valid, real, arms)
        (term.pow(2).sum() + term.sum()).backward()
        assert pt.grad is None or float(pt.grad.abs().sum()) == 0.0, "the prior path must not reach the keypoint positions"
        if step == 0:
            assert float(head.feat.weight.grad.abs().sum()) > 0 and float(sum(p.grad.abs().sum() for p in head.gate.parameters())) == 0.0, "step 1: the feature layer has a gradient, the gate not yet"
        opt.step()
        opt.zero_grad()
    assert all(not torch.equal(a, p) for a, p in zip(before, head.parameters())) and float(term.detach().abs().max()) > 0, "after a few steps every prior parameter has moved"
    print("jprior selftest ok; prior params", sum(p.numel() for p in PriorHead(64).parameters()))


if __name__ == "__main__":
    _selftest()
