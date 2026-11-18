"""Probing: where do the errors come from, and where is it worth looking? Works on a stored quantitative run, no network needed.

A Task that implements `probe()` stores for every frame, next to its counts, descriptors of the situation ('tag.*': a junction,
the curvature, the range, night ...) and the error assigned to causes ('cause.*', one unit that adds up: metres, cells, objects). From
these the Prober builds

  budget(run)              the error of the whole run split by cause (what it is, what to improve, and where to work: the place the
                           Task gives in causes(), e.g. model / code / evaluation; without one the cause name)
  situations(run)          groups of frames that share a property (a tag, a dominant cause, the worst decile, a plausibility flag) with
                           their share of the frames, their share of the error (excess = error share - frame share: the situations that
                           cost more than their size) and the headline scores inside / outside
  sample(run, situation)   random frames of a situation, drawn like everywhere in the suite: first a video (scene), then a frame in it
  scan(run)                budget + the situations that cost most + k random examples of each, drawn, as one html page: an impression of
                           where the errors come from, and for each example the command that opens it in the qualitative suite

Reading a result: a cause with a large share and a code / evaluation fix is the cheapest place to start (an error made by a rule of
the post-processing, or by ground truth that is simply missing there); a large share with a model fix is the research problem; a situation
with a large excess tells under which conditions it happens.
"""
from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from .contracts import Page, Panel
from .render import report as html
from .render.figure import render_page
from .store import RunStore


def place_of(cause: str, info: tuple) -> str:
    """where to work on a cause: the third entry of Task.causes()[cause] (e.g. 'model', 'code', 'evaluation', 'model/evaluation' = shared
    equally), supplied by the Task; without it the cause name itself (the core knows no words of any problem)"""
    return str(info[2]) if len(info) > 2 and info[2] else cause


def budget_panels(bud: List[Dict[str, Any]]) -> List[Panel]:
    """the bars of the error budget: missing (fn.<cause>) next to false (fp.<cause>) when the causes follow that convention (the toy and
    the lane task do), else one panel of all causes"""
    fn = {b["cause"][3:]: b["amount"] for b in bud if b["cause"].startswith("fn.") and b["amount"] > 0}
    fp = {b["cause"][3:]: b["amount"] for b in bud if b["cause"].startswith("fp.") and b["amount"] > 0}
    low_to_high = lambda d: dict(sorted(d.items(), key=lambda kv: kv[1])) or {"-": 0}                      # noqa: E731
    if fn or fp:
        return [Panel("bars", "missing (fn): amount per cause", low_to_high(fn)), Panel("bars", "false (fp): amount per cause", low_to_high(fp))]
    return [Panel("bars", "error: amount per cause", low_to_high({b["cause"]: b["amount"] for b in bud if b["amount"] > 0}))]


class Prober:
    def __init__(self, suite):
        self.s = suite
        self.dir = os.path.join(suite.out, "probe")

    # ------------------------------------------------------------------------------------------------ the data
    def frames(self, run: Union[str, RunStore]) -> List[Dict[str, Any]]:
        st = self.s.quant.open(run)
        rows = []
        for fr in st.frames:
            pr = (fr.get("info") or {}).get("probe") or {}
            causes = {k[6:]: float(v) for k, v in pr.items() if k.startswith("cause.")}
            rows.append({"label": fr["label"], "scene": fr["scene"], "index": fr["index"],
                         "tags": {k[4:]: float(v) for k, v in pr.items() if k.startswith("tag.")}, "causes": causes,
                         "error": float(sum(causes.values())), "scores": fr["scores"], "flags": [f["name"] for f in fr["flags"]]})
        if rows and not any(r["tags"] or r["causes"] for r in rows):
            raise ValueError("the run has no probe data: the Task does not implement probe() (or the run is older than that)")
        return rows

    def _key_scores(self, st: RunStore, rows) -> List[str]:
        keys: List[str] = []
        if not rows:
            return keys
        for name, m in st.manifest.get("metrics", {}).items():
            for k in m.get("headline", []):
                if f"{name}.{k}" in rows[0]["scores"]:
                    keys.append(f"{name}.{k}")
                    break
        return keys

    # ------------------------------------------------------------------------------------------------ budget
    def budget(self, run, labels: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
        rows = self.frames(run)
        want = None if labels is None else set(labels)
        sel = [r for r in rows if want is None or r["label"] in want]
        names = sorted({c for r in sel for c in r["causes"]})
        info = self.s.task.causes()
        tot = {c: sum(r["causes"].get(c, 0.0) for r in sel) for c in names}
        allv = sum(tot.values())
        out = []
        for c in sorted(names, key=lambda c: -tot[c]):
            what, fix = (tuple(info.get(c, ())) + ("", ""))[:2]
            out.append({"cause": c, "amount": tot[c], "share": tot[c] / allv if allv > 0 else 0.0, "frames": sum(r["causes"].get(c, 0.0) > 0 for r in sel),
                        "what": what, "fix": fix, "target": place_of(c, tuple(info.get(c, ())))})
        return out

    def targets(self, budget: List[Dict[str, Any]]) -> Dict[str, float]:
        """share of the error per place to work (a cause naming two places is split equally)"""
        out: Dict[str, float] = {}
        for b in budget:
            parts = b["target"].split("/") if b["target"] else ["unknown"]
            for p in parts:
                out[p] = out.get(p, 0.0) + b["share"] / len(parts)
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    # ------------------------------------------------------------------------------------------------ situations
    def situations(self, run, min_frames: int = 3, quantile: float = 0.75, dominant: float = 0.5) -> List[Dict[str, Any]]:
        st = self.s.quant.open(run)
        rows = self.frames(st)
        if not rows:                                         # an empty run has no situations
            return []
        n, E = len(rows), sum(r["error"] for r in rows)
        keys = self._key_scores(st, rows)
        cand: List[tuple] = []
        tags = sorted({t for r in rows for t in r["tags"]})
        for t in tags:
            v = np.array([r["tags"].get(t, np.nan) for r in rows])
            ok = np.isfinite(v)
            if not ok.any():
                continue
            if set(np.unique(v[ok])) <= {0.0, 1.0}:
                cand.append((f"{t}", "tag", f"{t} is true", ok & (v > 0.5)))
                continue
            lo, hi = np.nanquantile(v, 1 - quantile), np.nanquantile(v, quantile)
            if hi > lo:
                cand.append((f"{t} high", "tag", f"{t} >= {hi:.3g} (top quarter)", ok & (v >= hi)))
                cand.append((f"{t} low", "tag", f"{t} <= {lo:.3g} (bottom quarter)", ok & (v <= lo)))
        causes = sorted({c for r in rows for c in r["causes"]})
        for c in causes:
            cand.append((f"dominated by {c}", "cause", f"more than {dominant:.0%} of the frame's error is {c}",
                         np.array([r["error"] > 0 and r["causes"].get(c, 0.0) >= dominant * r["error"] for r in rows])))
        if keys:
            k0 = keys[0]
            better = self._direction(st, k0)
            v = np.array([r["scores"].get(k0, np.nan) for r in rows])
            ok = np.isfinite(v)
            if ok.any():
                thr = np.nanquantile(v, 0.1) if better == "high" else np.nanquantile(v, 0.9)
                cand.append((f"worst 10 % by {k0}", "score", f"{k0} {'<=' if better == 'high' else '>='} {thr:.3g}", ok & ((v <= thr) if better == "high" else (v >= thr))))
        for f in sorted({f for r in rows for f in r["flags"]}):
            cand.append((f"flag: {f}", "flag", f"plausibility flag '{f}' raised", np.array([f in r["flags"] for r in rows])))
        out = []
        for name, kind, desc, mask in cand:
            m = int(mask.sum())
            if m < min_frames or m == n:
                continue
            err = sum(r["error"] for r, k in zip(rows, mask) if k)
            share, eshare = m / n, (err / E if E > 0 else 0.0)
            row = {"name": name, "kind": kind, "desc": desc, "n": m, "share": share, "scenes": len({r["scene"] for r, k in zip(rows, mask) if k}),
                   "error_share": eshare, "excess": eshare - share, "labels": [r["label"] for r, k in zip(rows, mask) if k]}
            for k in keys:
                a = [r["scores"][k] for r, kk in zip(rows, mask) if kk and k in r["scores"] and np.isfinite(r["scores"][k])]
                b = [r["scores"][k] for r, kk in zip(rows, mask) if not kk and k in r["scores"] and np.isfinite(r["scores"][k])]
                row[f"in {k}"], row[f"out {k}"] = (float(np.mean(a)) if a else float("nan")), (float(np.mean(b)) if b else float("nan"))
            out.append(row)
        out.sort(key=lambda r: -r["excess"])
        return out

    def _direction(self, st: RunStore, key: str) -> str:
        name, k = key.split(".", 1)
        return st.manifest.get("metrics", {}).get(name, {}).get("better", {}).get(k, "high")

    # ------------------------------------------------------------------------------------------------ sampling
    def draw(self, labels: Sequence[str], k: int, seed: int = 0, scene_of: Optional[Dict[str, str]] = None) -> List[str]:
        """k frames out of `labels` like the suite draws everywhere: first a video (scene, every scene that has a candidate equally likely),
        then a frame in it; no frame twice"""
        rng = np.random.RandomState(seed)
        by: Dict[str, List[str]] = {}
        for lb in labels:
            by.setdefault(scene_of[lb] if scene_of else lb.rsplit("/", 1)[0], []).append(lb)
        out = []
        while len(out) < min(k, len(labels)):
            sc = sorted(by)[int(rng.randint(len(by)))]
            lb = by[sc].pop(int(rng.randint(len(by[sc]))))
            out.append(lb)
            if not by[sc]:
                del by[sc]
        return out

    def sample(self, run, situation: str, k: int = 6, seed: int = 0, render: bool = True) -> List[Dict[str, Any]]:
        sits = {s["name"]: s for s in self.situations(run, min_frames=1)}
        if situation not in sits:
            raise KeyError(f"unknown situation {situation!r}; known: {sorted(sits)}")
        rows = {r["label"]: r for r in self.frames(run)}
        labels = self.draw(sits[situation]["labels"], k, seed, {l: rows[l]["scene"] for l in sits[situation]["labels"]})
        out = []
        for lb in labels:
            r = rows[lb]
            d = {"label": lb, "scene": r["scene"], "error": r["error"], "top_cause": max(r["causes"], key=r["causes"].get) if r["error"] > 0 else "",
                 "tags": r["tags"], "command": f"quant trace {self.s.quant.open(run).manifest.get('name', run)} {lb}"}
            if render:
                d["images"] = self.s.quant.trace(run, lb, ("overview",)).images
            out.append(d)
        return out

    # ------------------------------------------------------------------------------------------------ scan
    def scan(self, run, k: int = 3, top: int = 6, seed: int = 0, min_frames: int = 3, render: bool = True) -> Dict[str, Any]:
        """budget + the `top` situations with the largest excess error + k random examples of each -> <out>/probe/<run>_s<seed>/index.html"""
        st = self.s.quant.open(run)
        name = st.manifest.get("name", os.path.basename(st.path))
        out_dir = os.path.join(self.dir, f"{name}_s{seed}")
        os.makedirs(out_dir, exist_ok=True)
        rows = self.frames(st)
        bud = self.budget(st)
        sits = [s for s in self.situations(st, min_frames=min_frames) if s["excess"] > 0][:top]
        by_label = {r["label"]: r for r in rows}
        tgt = self.targets(bud)
        sections = []
        sections.append(("what this is", html.table([["field", "value"], ["run", name], ["frames", len(rows)], ["scenes", len(st.scene_names())],
                                                      ["total error", f"{sum(r['error'] for r in rows):.0f} (units of the task's causes, e.g. metres)"],
                                                      ["seed", seed]], left_cols=(0, 1))))
        # budget
        bt = [["cause", "amount", "share", "frames", "what it is", "what to improve", "place"]] + [
            [b["cause"], b["amount"], f"{b['share']:.1%}", b["frames"], b["what"], b["fix"], b["target"]] for b in bud]
        body = html.table(bt, left_cols=(0, 4, 5, 6))
        if tgt:
            body += "<p>where to work (share of the error): " + ", ".join(f"<b>{k_}</b> {v:.0%}" for k_, v in tgt.items()) + "</p>"
        if bud:
            panels = budget_panels(bud)
            p = os.path.join(out_dir, "budget.png")
            render_page(Page(f"error budget of {name}", panels, ncols=len(panels), panel_size=(6.5, 3.6)), p)
            body += html.image(p, out_dir, 900)
        sections.append(("error budget: which cause the error is assigned to", body))
        # situations
        keys = [c[3:] for c in (sits[0] if sits else {}) if c.startswith("in ")]
        st_rows = [["situation", "kind", "frames", "of all", "scenes", "error share", "excess"] + [f"{k_} in / out" for k_ in keys]]
        for s in sits:
            st_rows.append([s["name"], s["kind"], s["n"], f"{s['share']:.0%}", s["scenes"], f"{s['error_share']:.0%}", f"{s['excess']:+.0%}"] +
                           [f"{s['in ' + k_]:.3g} / {s['out ' + k_]:.3g}" for k_ in keys])
        sections.append(("situations that cost more than their size (excess = share of the error - share of the frames)", html.table(st_rows, left_cols=(0, 1))))
        draws: Dict[str, Any] = {}
        for s in sits:
            labels = self.draw(s["labels"], k, seed, {l: by_label[l]["scene"] for l in s["labels"]})
            cells = []
            draws[s["name"]] = []
            for lb in labels:
                r = by_label[lb]
                top_c = max(r["causes"], key=r["causes"].get) if r["error"] > 0 else "-"
                cap = f"{lb}: error {r['error']:.1f}, mostly {top_c}; " + ", ".join(f"{t}={v:.3g}" for t, v in list(r["tags"].items())[:5])
                cmd = f"quant trace {name} {lb}"
                if render:
                    img = self.s.quant.trace(st, lb, ("overview",)).images.get("overview")
                    cells.append(html.image(img, out_dir, 640, cap) + f"<div class=note><code>{cmd}</code></div>")
                else:
                    img = None
                    cells.append(f"<div class=note>{cap} <code>{cmd}</code></div>")
                draws[s["name"]].append({"label": lb, "error": r["error"], "top_cause": top_c, "image": img, "command": cmd})
            sections.append((f"{s['name']} - {s['desc']} ({s['n']} frames, {s['error_share']:.0%} of the error)", "".join(cells)))
        path = html.write(os.path.join(out_dir, "index.html"), f"probe: {name}", sections)
        with open(os.path.join(out_dir, "budget.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["cause", "amount", "share", "frames", "what", "fix", "place"])
            w.writerows([[b["cause"], round(b["amount"], 3), round(b["share"], 4), b["frames"], b["what"], b["fix"], b["target"]] for b in bud])
        with open(os.path.join(out_dir, "situations.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["situation", "kind", "frames", "share", "scenes", "error_share", "excess"])
            w.writerows([[s["name"], s["kind"], s["n"], round(s["share"], 4), s["scenes"], round(s["error_share"], 4), round(s["excess"], 4)]
                         for s in self.situations(st, min_frames=min_frames)])
        with open(os.path.join(out_dir, "draws.json"), "w", encoding="utf-8") as f:
            json.dump({"seed": seed, "k": k, "draws": draws, "targets": tgt}, f, indent=1)
        return {"index": path, "dir": out_dir, "budget": bud, "targets": tgt, "situations": sits, "draws": draws}
