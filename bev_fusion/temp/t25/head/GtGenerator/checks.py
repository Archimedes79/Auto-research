"""Automatic consistency checks of the generated GT over keyframes: python -m autogt check [--stride N] [--holdout]

Each check is a physical or bookkeeping fact the GT must obey; the report counts violations. `Checks` is an accumulator: feed it
the keyframes one by one (the `report` command feeds it while it builds the cache, so every keyframe is generated once).
Objects
  O1 box bottom on the ground: z of the box bottom in the ego frame (|z| > 0.5 m within 40 m = slope / overpass, a ground-plane
     assumption of the camera lift, not a GT error)
  O2 moving vehicles (> 1 m/s) stand on the drivable area
  O3 no two observed vehicle footprints overlap (IoU > 0.10)
  O4 boxes without lidar / radar points (ignored): share and range
  O5 our occlusion mask against the annotators' visibility (should fall monotonically with visibility)
  O6 timing: camera time minus annotation time, and the shift v * dt applied to the moving boxes (unknown velocity: kept)
  O7 instance layer points at an observed box of the group whose layer covers the cell
  O8 stationary boxes keep their world pose between keyframes
Map
  M1 the ego car stands on the drivable area; M2 centre-line cells and M3 lane-divider cells lie on the drivable area
Velocity / flow
  V1 flow = flow_ego + velocity everywhere; velocity is 0 outside observed boxes; weight 0 on ignored boxes
  V2 flags of the velocity estimate over all boxes (radar_conflict, radar_override, stationary_but_moving, short_track)
  V3 radar field of view: share of all radar detections inside +-60 deg of the sensor axis (the assumption of the radar mask)
--holdout: leave-one-radar-out test of the radar correction: each radar is scored on boxes corrected with the OTHER four
  (radial error against the held-out Doppler: RMSE / median / within 0.5 m/s; track only vs corrected vs zero velocity)
"""
import collections
import hashlib
import math
import os

import numpy as np
from shapely.geometry import Polygon

from . import flow as flowmod
from .grid import GridSpec
from .nuscenes_adapter import RADARS, NuScenesScene
from .suite import GtGenerator


SRC_HASH = hashlib.sha1(open(__file__, "rb").read()).hexdigest()[:10]       # version of the checks: stored in every accumulator


def _cell(grid, x, y):
    i, j = int((x - grid.x_min) / grid.res), int((y - grid.y_min) / grid.res)
    return (i, j) if 0 <= i < grid.nx and 0 <= j < grid.ny else None


class Checks:
    """accumulator of the checks above; feed(scene_name, k, token, gt_dict, annotation_time_us) per keyframe, then text() /
    scene_rows(); small and picklable (workers return it, `merge` adds another one)."""

    def __init__(self, grid):
        self.grid, self.st = grid, collections.defaultdict(list)
        self.version = SRC_HASH
        self.flags = collections.Counter()
        self.prev_world, self.scene = {}, None
        self.per_scene = collections.OrderedDict()

    def feed(self, name, k, tok, gk, sample_time_us):
        g, st = self.grid, self.st
        if name != self.scene:
            self.scene, self.prev_world = name, {}
            self.per_scene[name] = collections.Counter()
        ps = self.per_scene[name]
        m, da = gk["meta"], gk["drivable_area"]
        st["dt_ms"].append((m["timestamp"] - sample_time_us) * 1e-3)
        ps["keyframes"] += 1
        for b in m["boxes"]:
            x, y, z = b["center"]
            self.flags[b["vel_flag"]] += 1
            if b["vel_flag"] in ("radar_override", "radar_conflict", "short_track", "stationary_but_moving"):
                ps[b["vel_flag"]] += 1
            if not (0 < x < 60 and abs(y) < 26):
                continue
            if not b["observed"]:
                st["O4_unobs_range"].append(math.hypot(x, y))
                ps["ignored"] += 1
                continue
            st["O4_obs"].append(1)
            ps["observed"] += 1
            ps["stationary"] += b["stationary"]
            sp = math.hypot(*b["vel"])
            ps["moving"] += (not b["stationary"]) and sp > 1.0
            zb = z - b["size"][2] / 2
            if x < 40:
                st["O1_z"].append(zb)
            c = _cell(g, x, y)
            if b["group"] == "vehicle" and sp > 1 and c:
                st["O2"].append(da[c] > 0)
            if sp > 1 and b.get("shift_s", 0.0) != 0.0:
                st["O6_shift_cm"].append(math.hypot(*b["shift"]) * 100)
            if not b["stationary"] and not b["vel_known"]:
                st["O6_kept"].append(1)                          # unknown velocity: stays at the annotation time
            if c and gk["camera"][c] > 0:
                st[f"O5_vis{b['visibility']}"].append(gk["occluded"][c] > 0)
            if b["stationary"]:
                if b["instance"] in self.prev_world:
                    st["O8_move_cm"].append(np.linalg.norm(self.prev_world[b["instance"]] - b["world"]) * 100)
                self.prev_world[b["instance"]] = b["world"]
        vb = [Polygon(b["corners"]) for b in m["boxes"] if b["group"] == "vehicle" and b["observed"]]
        for a in range(len(vb)):
            for c2 in range(a + 1, len(vb)):
                if vb[a].intersects(vb[c2]) and vb[a].intersection(vb[c2]).area / max(vb[a].union(vb[c2]).area, 1e-6) > 0.10:
                    st["O3"].append(1)
        ii = gk["instance"]
        for i in np.unique(ii[ii >= 0]):
            st["O7"].append(m["boxes"][i]["observed"] and gk[m["boxes"][i]["group"]][ii == i].min() > 0)
        st["M1"].append(float(da[0:6, 50:54].mean()))
        ps["drivable"] += float(da.mean())
        if (gk["centreline"] > 0).any():
            st["M2"].append(float((da[gk["centreline"] > 0] > 0).mean()))
        if (gk["lane_divider"] > 0).any():
            st["M3"].append(float((da[gk["lane_divider"] > 0] > 0).mean()))
        st["V1_flow"].append(float(np.abs(gk["flow"] - gk["flow_ego"] - gk["velocity"]).max()))
        obj = gk["vehicle"] + gk["pedestrian"] + gk["two_wheeler"]
        st["V1_vel_outside"].append(float(np.abs(gk["velocity"][:, obj == 0]).max()) if (obj == 0).any() else 0.0)
        st["V1_w"].append(bool((gk["flow_weight"][gk["ignore"] > 0.9] == 0).all()))

    def merge(self, other):
        """add the accumulated results of another Checks (e.g. of a worker that did other sequences)."""
        for k, v in other.st.items():
            self.st[k].extend(v)
        self.flags.update(other.flags)
        for n, c in other.per_scene.items():
            self.per_scene[n] = c
        return self

    def text(self, fov=None):
        """the report lines (a list of str)."""
        st = self.st
        rate = lambda v: f"{np.mean(v):.1%} of {len(v)}"  # noqa: E731
        dt = np.array(st["dt_ms"])
        if not len(dt):
            return ["no keyframe was checked (every sequence failed or none was asked for)"]
        z, sh, ur = np.array(st["O1_z"]), np.array(st["O6_shift_cm"]), np.array(st["O4_unobs_range"])
        out = [f"keyframes {len(dt)} in {len(self.per_scene)} sequences; camera minus annotation time: median {np.median(dt):+.1f} ms, "
               f"range {dt.min():+.1f}..{dt.max():+.1f}",
               f"O1 box bottom z (< 40 m): median {np.median(z):+.2f} m, |z| > 0.5 m in {np.mean(np.abs(z) > 0.5):.1%} of {len(z)}",
               f"O2 moving vehicles on the drivable area: {rate(st['O2'])}",
               f"O3 overlapping observed vehicle pairs (IoU > 0.10): {len(st['O3'])}",
               f"O4 boxes without lidar / radar points: {len(ur)} vs {len(st['O4_obs'])} observed (median range "
               f"{np.median(ur) if len(ur) else float('nan'):.0f} m)",
               "O5 centre cell in our occlusion mask by annotator visibility 0-40 / 40-60 / 60-80 / 80-100 %: " +
               " / ".join(f"{np.mean(st[f'O5_vis{v}']):.0%}" if st[f"O5_vis{v}"] else "-" for v in "1234"),
               f"O6 moving boxes (> 1 m/s) moved from the annotation to the camera time by v * dt: {len(sh)}, shift median "
               f"{np.median(sh) if len(sh) else float('nan'):.0f} cm, 95 % {np.percentile(sh, 95) if len(sh) else float('nan'):.0f} cm, "
               f"max {sh.max() if len(sh) else float('nan'):.0f} cm; observed boxes of unknown velocity kept at the annotation time: "
               f"{len(st['O6_kept'])}",
               f"O7 instance layer consistent: {rate(st['O7'])}",
               f"O8 stationary boxes, world motion between keyframes: max {max(st['O8_move_cm'], default=0):.2f} cm (n = {len(st['O8_move_cm'])})",
               f"M1 ego on the road: median {np.median(st['M1']):.2f}, < 0.5 in {np.mean(np.array(st['M1']) < 0.5):.1%}; "
               f"M2 centre-line cells on the road {np.median(st['M2']):.1%}; M3 lane-divider cells on the road {np.median(st['M3']):.1%}",
               f"V1 flow = flow_ego + velocity (max abs difference {max(st['V1_flow']):.1e}); velocity outside boxes (max "
               f"{max(st['V1_vel_outside']):.1e}); weight 0 on ignored boxes: {rate(st['V1_w'])}",
               f"V2 velocity flags over all boxes: {dict(self.flags)}"]
        if fov is not None:
            out.append(f"V3 radar field of view: {fov:.2%} of all detections lie within +-60 deg of the sensor axis")
        return out

    def scene_rows(self):
        """per sequence: (name, keyframes, observed boxes, moving, stationary, ignored, overrides + conflicts, drivable share)."""
        return [(n, c["keyframes"], c["observed"], c["moving"], c["stationary"], c["ignored"],
                 c["radar_override"] + c["radar_conflict"], c["drivable"] / max(c["keyframes"], 1)) for n, c in self.per_scene.items()]


def _feed(h, x):
    if isinstance(x, np.ndarray):
        h.update(f"A{x.dtype}{x.shape}".encode())
        h.update(np.ascontiguousarray(x).tobytes())
    elif isinstance(x, dict):
        for key in sorted(x, key=str):
            h.update(f"K{key}".encode())
            _feed(h, x[key])
    elif isinstance(x, (list, tuple)):
        h.update(f"L{len(x)}".encode())
        for v in x:
            _feed(h, v)
    else:
        h.update(f"{type(x).__name__}{x!r}".encode())


def digests(dataroot, version="v1.0-mini", stride=2, sweeps=("scene-0061", "scene-1077", "scene-0796")):
    """one digest per frame (every array byte and every meta entry) of every `stride`-th keyframe of all sequences and of a few 12 Hz sweep
    frames: the evidence that two generators make the same GT (`same_gt_as` in versions.csv). Run `python -m autogt digest OUT.json` in both
    source trees and compare the files (`digest --compare A.json B.json`). -> {signature, digests: {token or token@k: sha1}}"""
    dataroot = os.path.expanduser(dataroot)
    gen = GtGenerator(dataroot, version, light=True)
    from .suite import scene_tokens
    out = {}
    for name in sorted({s["name"] for s in gen.nusc.scene}):
        for tok in scene_tokens(name, version, dataroot)[::stride]:
            h = hashlib.sha1()
            _feed(h, gen.keyframe(tok))
            out[tok] = h.hexdigest()
    for name in sweeps:
        tok = scene_tokens(name, version, dataroot)[8]
        track = gen.camera_track(tok)
        i = [x["token"] for x in track].index(gen.cam(tok)["token"])
        for j in (1, 3):
            h = hashlib.sha1()
            _feed(h, gen.keyframe_gt(tok, GridSpec(), sd=track[i + j]))
            out[f"{tok}@{j}"] = h.hexdigest()
    return {"signature": gen.signature(), "digests": out}


def run(dataroot, version="v1.0-mini", scenes=None, stride=1, holdout=False):
    gen = GtGenerator(dataroot, version)
    ck = Checks(gen.grid)
    for name in (scenes or gen.scenes()):
        for k, tok in enumerate(gen.keyframes(name)):
            if k % stride == 0:
                ck.feed(name, k, tok, gen.keyframe(tok), gen.nusc.get("sample", tok)["timestamp"])
        print(name, flush=True)
    print("\n=== GT checks ===")
    print("\n".join(ck.text(radar_fov(gen, scenes, stride))))
    if holdout:
        print("\n".join(leave_one_radar_out(gen, scenes, stride)))
    return ck


def radar_fov(gen, scenes=None, stride=1):
    from nuscenes.utils.data_classes import RadarPointCloud
    inside = total = 0
    for name in (scenes or gen.scenes())[:5]:
        for k, tok in enumerate(gen.keyframes(name)):
            if k % max(stride, 4):
                continue
            for ch, sdt in gen.nusc.get("sample", tok)["data"].items():
                if ch.startswith("RADAR"):
                    sd = gen.nusc.get("sample_data", sdt)
                    pts = RadarPointCloud.from_file(os.path.join(gen.dataroot, sd["filename"])).points
                    ang = np.abs(np.degrees(np.arctan2(pts[1], pts[0])))
                    inside += int((ang <= 60).sum())
                    total += pts.shape[1]
    return inside / max(total, 1)


def leave_one_radar_out(gen, scenes=None, stride=1):
    """the leave-one-radar-out test of the radar correction; returns the report lines."""
    res = collections.defaultdict(list)
    per_pair = []
    fl = collections.Counter()
    for name in (scenes or gen.scenes()):
        sc = NuScenesScene(gen.nusc, gen.dataroot, name)
        by_id = {tr.id: tr for tr in sc.tracks}
        for k, tok in enumerate(gen.keyframes(name)):
            if k % stride:
                continue
            t = sc.time_of(tok, "annotation")
            frames = sc.radar(t)
            prior = {o.id: o for o in flowmod.estimate_flow(sc.tracks, sc.ego, t).objects}
            for R in RADARS:
                held = [f for f in frames if f.sensor == R]
                rest = [f for f in frames if f.sensor != R]
                if not held:
                    continue
                fused = {o.id: o for o in flowmod.estimate_flow(sc.tracks, sc.ego, t, rest).objects}
                for i, op in prior.items():
                    if op.stationary or not op.observed or not np.isfinite(op.v).all():
                        continue
                    rh, d, _ = flowmod.radar_points_in_box(op, by_id[i], held, 0.5)
                    of = fused[i]
                    if len(d) == 0 or max(op.speed, of.speed) < 1.0:
                        continue
                    fl[of.flag] += 1
                    res["zero"].extend((-d).tolist())
                    res["track"].extend((rh @ op.v - d).tolist())
                    res["radar-corrected"].extend((rh @ of.v - d).tolist())
                    per_pair.append((float(np.sqrt(np.mean((rh @ op.v - d) ** 2))), float(np.sqrt(np.mean((rh @ of.v - d) ** 2)))))
        print(name, "(holdout)", flush=True)
    lines = [f"leave-one-radar-out: {len(res['zero'])} held-out detections in moving observed boxes ({len(per_pair)} box-radar pairs)",
             "estimate | RMSE (m/s) | median |err| | within 0.5 m/s | share > 2 m/s"]
    for k in ("zero", "track", "radar-corrected"):
        e = np.abs(np.array(res[k]))
        lines.append(f"{k:15s} | {np.sqrt((e ** 2).mean()):.3f} | {np.median(e):.3f} | {np.mean(e < 0.5):.1%} | {np.mean(e > 2):.1%}")
    if not len(per_pair):
        return lines + ["per box-radar pair: none in this selection"]
    pp = np.array(per_pair)
    diff = pp[:, 1] - pp[:, 0]
    rng = np.random.default_rng(0)
    boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
    lines.append(f"per box-radar pair: corrected better {np.mean(diff < -1e-9):.1%}, equal {np.mean(np.abs(diff) <= 1e-9):.1%}, "
                 f"worse {np.mean(diff > 1e-9):.1%}; mean RMSE change {diff.mean():+.3f} m/s (bootstrap {np.percentile(boot, 2.5):+.3f} .. "
                 f"{np.percentile(boot, 97.5):+.3f}; pairs of one scene are correlated: optimistic); flags {dict(fl)}")
    return lines
