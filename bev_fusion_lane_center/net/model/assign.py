"""Grid stage 4: soft assignment of the sparse lane vertices to the SD-map roads and lanes (E2E, a softmax with a dustbin; no matching, no argmax).

Idea of the pose-free part is copied from the lane project (training/data/assoc_lib.py assign_points: nearest map point with a matching direction within r; and
sd_lib.py: the SD map = roads with a lane count), made soft: every vertex i (position, heading, weight w_i from the sparse map) gets a distribution over the map
tokens j and a dustbin,  A_ij = softmax_j( -[ |p_i - q_j|^2 / sigma^2 + lam (1 - h_i . t_j) ]  |  -c_bin ).  Three learned scalars (sigma, lam, c_bin).
The rough localisation of the map (a pose error of metres) is NOT in here: it is the existing differentiable `training/models/p18_regis.py` SoftRegister that runs
before (the map is shifted by its estimate); this module sees the corrected map.

Outputs per vertex:  A (N, M+1) over map tokens + dustbin;  P_road (N, R+1) over roads + unassigned (sum of A over the tokens of a road);
                     lat (N,) signed lateral offset to the assigned map line (left positive), nlane (N,) its lane count; P_slot (N, K) over the lane slots of the road.
Per road:  mass_r = sum_i w_i P_road[i, r]  (how much perceived lane there is on road r) and  present_r = 1 - exp(-mass_r / expected_r)  (a confidence in [0, 1]).
The topology of the delivered graph is the MAP's (the road links of the SD map), the perception supplies where each lane really is and whether it exists.

Conventions as soft_lines.py: x forward, y left. `map_tokens` takes an SD map of sd_lib (polylines in the metric code's frame x right, y forward) and converts.
Run the self-test: python assign.py
"""
import math

import numpy as np
import torch
import torch.nn as nn

LANE_W = 3.5


def _resample(p, ds):
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.linspace(0.0, s[-1], max(int(np.ceil(s[-1] / ds)), 1) + 1)
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], 1), t


def _tangents(q):
    t = np.gradient(q, axis=0)
    return t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)


def map_tokens(sd, ds=2.0, convert=True):
    """SD map dict(roads=[(poly, n)], route=[(poly, n)]) -> dict of numpy arrays pos (M, 2), dir (M, 2), n (M,), rid (M,), route (M,) bool, s (M,) arclength in the road;
    a road is a token id 0..R-1, the route pieces follow as ids R.. (flag route); convert: (x right, y forward) -> (x forward, y left)"""
    pos, dr, nn_, rid, rte, ss = [], [], [], [], [], []
    k = 0
    for key in ("roads", "route"):
        for poly, cnt in sd[key]:
            p = np.asarray(poly, np.float64)
            if len(p) < 2 or np.linalg.norm(np.diff(p, axis=0), axis=1).sum() < 1e-6:
                k += 1
                continue
            q, s = _resample(p, ds)
            c = np.interp(s, np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]), np.asarray(cnt, np.float64))
            t = _tangents(q)
            if convert:
                q, t = q[:, ::-1] * np.array([1.0, -1.0]), t[:, ::-1] * np.array([1.0, -1.0])
            pos.append(q), dr.append(t), nn_.append(c), rid.append(np.full(len(q), k)), rte.append(np.full(len(q), key == "route")), ss.append(s)
            k += 1
    cat = lambda a, dt: np.concatenate(a).astype(dt) if a else np.zeros((0,), dt)  # noqa: E731
    return dict(pos=cat(pos, np.float32).reshape(-1, 2), dir=cat(dr, np.float32).reshape(-1, 2), n=cat(nn_, np.float32), rid=cat(rid, np.int64),
                route=cat(rte, bool), s=cat(ss, np.float32), n_ids=k)


class Assigner(nn.Module):
    def __init__(self, sigma=1.5, lam=2.0, c_bin=4.0, sigma_lat=0.8, k_slots=4):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.tensor(math.log(sigma)))
        self.lam = nn.Parameter(torch.tensor(float(lam)))
        self.c_bin = nn.Parameter(torch.tensor(float(c_bin)))
        self.sigma_lat = sigma_lat
        self.k_slots = k_slots

    def forward(self, pos, hdg, w, mp):
        """pos (N, 2), hdg (N, 2) unit, w (N,) weights; mp = tensors of map_tokens (pos (M,2), dir (M,2), n (M,), rid (M,)) -> dict"""
        N, M = pos.shape[0], mp["pos"].shape[0]
        R = int(mp["n_ids"])
        d = pos[:, None, :] - mp["pos"][None, :, :]                                # (N, M, 2)
        d2 = (d * d).sum(-1)
        cost = d2 / torch.exp(2 * self.log_sigma) + self.lam * (1.0 - hdg @ mp["dir"].T)
        A = torch.softmax(-torch.cat([cost, self.c_bin.expand(N, 1)], 1), 1)       # (N, M+1)
        P_road = torch.zeros(N, R + 1, dtype=A.dtype, device=A.device).index_add_(1, mp["rid"], A[:, :M])
        P_road[:, R] = A[:, M]                                                     # the dustbin = no road
        a = A[:, :M] / A[:, :M].sum(1, keepdim=True).clamp_min(1e-6)               # inside the map (normalised without the dustbin)
        nrm = torch.stack([-mp["dir"][:, 1], mp["dir"][:, 0]], -1)                 # left normal of the map line
        lat = (a * (d * nrm[None]).sum(-1)).sum(1)
        nlane = (a * mp["n"][None]).sum(1)
        k = torch.arange(self.k_slots, device=pos.device, dtype=pos.dtype)
        centre = (k[None] - (nlane[:, None] - 1) / 2) * LANE_W                     # lane slot centres, the map line sits in the middle of the carriageway
        logit = -((lat[:, None] - centre) ** 2) / (2 * self.sigma_lat ** 2)
        logit = logit.masked_fill(k[None] >= torch.round(nlane)[:, None], -1e4)
        P_slot = torch.softmax(logit, 1)
        mass = (w[:, None] * P_road[:, :R]).sum(0)                                 # (R,)
        return dict(A=A, P_road=P_road, lat=lat, nlane=nlane, P_slot=P_slot, mass=mass)


def present(mass, length_m, per_100m=11.0):
    """confidence in [0, 1] that road r is perceived: 1 - exp(-mass / expected vertex mass on its visible length)"""
    return 1.0 - torch.exp(-mass / (per_100m * length_m / 100.0 + 1.0))


def to_torch(mp, device="cpu"):
    out = {k: torch.as_tensor(v, device=device) for k, v in mp.items() if isinstance(v, np.ndarray)}
    out["n_ids"] = mp["n_ids"]
    return out


def _selftest():
    torch.manual_seed(0)
    np.random.seed(0)
    # a map in the metric code's frame (x right, y forward): road 0 straight ahead (2 lanes), road 1 a crossing road, road 2 curved (1 lane)
    sd = dict(roads=[(np.array([[0.0, -50.0], [0.0, 50.0]]), np.array([2.0, 2.0])),
                     (np.array([[-50.0, 20.0], [50.0, 20.0]]), np.array([1.0, 1.0])),
                     (np.array([[8.0, -50.0], [12.0, 0.0], [30.0, 50.0]]), np.array([1.0, 1.0, 1.0]))], route=[])
    mp = map_tokens(sd)
    assert set(np.unique(mp["rid"])) == {0, 1, 2} and mp["pos"].shape[0] > 100
    assert np.allclose(mp["dir"][mp["rid"] == 0], [1.0, 0.0], atol=1e-5), "road 0 runs straight ahead: heading (1, 0) in (x forward, y left)"
    t = to_torch(mp)
    # perceived vertices: on road 0 (a lane 1.75 m to the left of its centre line) with 0.4 m noise, plus a distractor line 14 m beside everything, heading across
    n0 = 60
    x = torch.linspace(-45, 45, n0)
    p0 = torch.stack([x, torch.full_like(x, 1.75)], 1) + 0.4 * torch.randn(n0, 2)
    h0 = torch.nn.functional.normalize(torch.tensor([1.0, 0.0]) + 0.05 * torch.randn(n0, 2), dim=1)
    p1 = torch.stack([torch.linspace(-40, 40, 20), torch.full((20,), -14.0)], 1)   # nothing in the map there
    h1 = torch.tensor([0.0, 1.0]).expand(20, 2)
    pos, hdg = torch.cat([p0, p1]).requires_grad_(True), torch.cat([h0, h1]).requires_grad_(True)
    w = torch.ones(n0 + 20, requires_grad=True)
    mod = Assigner()
    out = mod(pos, hdg, w, t)
    acc = (out["P_road"][:n0].argmax(1) == 0).float().mean()
    dust = out["P_road"][n0:, 3].mean()
    print("road accuracy", float(acc), "| distractor in the dustbin", float(dust), "| lateral offset", float(out["lat"][:n0].mean()),
          "| lanes", float(out["nlane"][:n0].mean()), "| slot", out["P_slot"][:n0].mean(0).tolist())
    assert acc > 0.95 and dust > 0.8, "assignment"
    assert abs(float(out["lat"][:n0].mean()) - 1.75) < 0.5, "the lateral offset of the lane beside the centre line"
    assert out["P_slot"][:n0].mean(0)[1] > 0.5, "the left lane of a two-lane road is slot 1"
    pr = present(out["mass"], torch.tensor([100.0, 100.0, 100.0]))
    print("present per road", [round(float(v), 3) for v in pr], "(road 2 runs 4 - 12 m beside the perceived lane over part of its length: some leak is right)")
    assert pr[0] > 0.9 and pr[1] < 0.2 and pr[0] > 3 * max(float(pr[1]), float(pr[2])), pr
    (out["A"].sum() * 0 + out["mass"][0] + out["lat"].sum() + out["P_slot"][:, 1].sum()).backward()
    for name, g in (("pos", pos.grad), ("hdg", hdg.grad), ("w", w.grad), ("sigma", mod.log_sigma.grad), ("lam", mod.lam.grad), ("c_bin", mod.c_bin.grad)):
        assert g is not None and float(g.abs().sum()) > 0, f"no gradient to {name}"
    print("assign selftest ok")


if __name__ == "__main__":
    _selftest()
