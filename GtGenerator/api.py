"""The interface of the suite: the methods of the class `GtGenerator` (the class itself is in suite.py; its static methods are these). The command line (`cli.py`) reaches the suite ONLY through them:
every command is one method, `python -m GtGenerator generate-bev ROOT` is `GtGenerator.generate_bev_gt(ROOT)`.

    from GtGenerator import GtGenerator
    key = GtGenerator.generate_bev_gt("~/data/nuscenes_full", set_name="full")        # front GT + lane GT of the rig bev_fusion + packed masks      -> GtKey
    key = GtGenerator.generate_lane_gt("~/data/nuscenes_full", rig="lane_center")      # the lane GT of Spurmitte (rig="all": both)                    -> GtKey (a dict of them for "all")
    key = GtGenerator.gt_key("bev_fusion", set_name="full")                            # what is stored, nothing is generated; key.complete, key.load(token)
    GtGenerator.generate(root, set_name="small")                                       # everything (front, both lane GTs, masks)                      -> dict, with the keys
    GtGenerator.have(set_name="full")  /  GtGenerator.check_data(root)  /  GtGenerator.audit_lane("bev_fusion")  /  GtGenerator.sets()  /  GtGenerator.versions()   -> dict {ok, text, ...}

The rules (the convention of the suites, autoresearch/README.md "The interface of the suites"): plain data in and out (paths, names, numbers, lists; a dict back, a `GtKey` for the keys); a method
that has nothing to say about progress does not print; a wrong argument raises ValueError with the message; a dict may hold `text` (what the command line prints) and `ok` (False: the thing
asked about is not there / not complete: exit code 1). Long runs print their progress when `verbose` (default True). The work itself is in the modules below this file (pipeline, gt_api,
store_audit, lane_raster, sets, checks, report, view, video, clean ...): these methods only shape the arguments and the result, so a module can be used without the class too.
"""
import contextlib
import json
import os


@contextlib.contextmanager
def _bad_pick():
    """an unknown sequence (StopIteration) or a keyframe number out of range (IndexError) in the picture commands is a wrong argument: ValueError with the message"""
    try:
        yield
    except (StopIteration, IndexError) as e:
        raise ValueError(f"no such sequence or keyframe ({type(e).__name__}): check the scene name and the keyframe number") from None


def _options(use_flow=True, use_radar=True, boxes_at="camera"):
    """the generator options of the command line and the methods (flow and radar on unless switched off; boxes_at only when it is not the default)"""
    return {"use_flow": use_flow, "use_radar": use_radar, **({} if boxes_at == "camera" else {"boxes_at": boxes_at})}


def _versions_text():
    """the table of versions.csv (key, signature, date, CURRENT for this source, stored when its folder exists), the lane versions and the releases"""
    from . import lane_gt, versions
    from .suite import DATA, code_signature
    out = []
    cur, rows = code_signature(), versions.table()
    for r in rows:
        same = f" (the GT of {r['same_gt_as']})" if r.get("same_gt_as") else ""
        out.append(f"{r['version']:3s} {r['signature']}  {r['date']}  {'CURRENT' if r['signature'] == cur else '       '}  "
                   f"{'stored    ' if os.path.isdir(os.path.join(DATA, versions.canonical(r['signature']))) else 'not stored'}  {r['tag']:11s} "
                   f"{r['what_changed'][:70]}{same}")
    if cur not in {r["signature"] for r in rows}:
        out.append(f"this source has the signature {cur}: no row in GtGenerator/versions.csv (unreleased): `python -m GtGenerator versions --update` derives it from git")
    here = versions.current_release()
    out.append("lane GT (own versions, GtGenerator/lane_versions.csv):")
    for r in lane_gt.table():
        same = f" (the GT of {r['same_gt_as']})" if r["same_gt_as"] else ""
        out.append(f"  {r['version']:3s} {r['rig']:11s} {r['signature']}  {r['date']}  {'CURRENT' if r['signature'] == lane_gt.code_signature(rig=r['rig']) else '       '}  "
                   f"{'stored    ' if os.path.isdir(os.path.join(lane_gt.DATA, lane_gt.store_of(r['signature']))) else 'not stored'}  {r['what_changed'][:60]}{same}")
    out.append("releases (the suite as one number, GtGenerator/releases.csv):")
    for r in versions.releases():
        out.append(f"  {r['release']:5s} front {r['front']:4s} + lane {r['lane']:3s}  {r['date']}  {'CURRENT' if r['release'] == here else '       '}  {r['what_changed'][:80]}")
    if here is None:
        out.append(f"this source (front {versions.version_of(cur)}, lane {lane_gt.version_of(lane_gt.code_signature())}) has no row in GtGenerator/releases.csv: `python -m GtGenerator versions --update` derives it")
    return "\n".join(out)


def _sets_text():
    """each scene set: its sequences, keyframes, and how much of it the store of the current generator holds"""
    from . import sets
    out = []
    for name in sets.names():
        try:
            s = sets.status(name)
        except ValueError as e:
            out.append(f"{name:6s} {len(sets.scenes(name)):4d} scenes  not readable here: {e}")
            continue
        out.append(f"{name:6s} {len(sets.scenes(name)):4d} scenes  {s['total']:6d} keyframes ({s['version']})  stored for this generator: "
                   f"{s['have']}/{s['total']}{'  COMPLETE' if s['complete'] else ''}")
    return "\n".join(out)


# parts of the suite that `selftest(only=...)` can run alone: name -> module
SELFTESTS = {"cli": "cli_selftest", "gt_api": "gt_api_selftest", "generator": "generator_selftest", "lane_graph": "lane_graph_selftest", "lane_tiers": "lane_tiers_selftest",
             "clean": "clean", "rig_masks": "rig_masks", "lane_toy": "lane_tiers", "jobs": "jobs_selftest"}


class Interface:
    """the static methods of `GtGenerator`: one per command of the command line, grouped as the README says (make GT / look at GT / check / pictures / tidy)"""

    # ------------------------------------------------------------------------------------------------------------ make GT
    @staticmethod
    def generate(dataroot, out=None, version=None, scenes=None, set_name=None, stride=1, res=0.5, workers=1, lane=True, verbose=True, use_flow=True, use_radar=True, boxes_at="camera"):
        """everything: the front GT, and (stride 1, lane=True) the lane GT of both systems and the packed masks, resumable (pipeline.generate)
        -> pipeline.generate's dict + `keys` {system: GtKey} (empty without the lane part) + `ok` (False when the derived masks failed)"""
        from . import gt_api, pipeline
        opts = _options(use_flow, use_radar, boxes_at)
        r = pipeline.generate(dataroot, out, version, scenes, stride, res, workers=workers, set_name=set_name, lane=lane, verbose=verbose, **opts)
        keys = {}
        if r["lane"] is not None:
            name = set_name or None
            keys = {system: gt_api.key_of(system, r["version"], name, out, dataroot=dataroot, scenes=scenes, options=opts) for system in ("bev_fusion", "lane_center")}
        return {**r, "keys": keys, "ok": not r["masks_error"]}

    @staticmethod
    def generate_bev_gt(*args, **kwargs):
        """BEV Fusion: the front GT + the lane GT of the rig bev_fusion + the packed area masks -> GtKey (gt_api.generate_bev_gt)"""
        from . import gt_api
        return gt_api.generate_bev_gt(*args, **kwargs)

    @staticmethod
    def generate_lane_gt(*args, **kwargs):
        """the lane GT of one system (rig lane_center | bev_fusion) or both (rig="all") -> GtKey, or a dict of them for "all" (gt_api.generate_lane_gt)"""
        from . import gt_api
        return gt_api.generate_lane_gt(*args, **kwargs)

    @staticmethod
    def area_masks(out=None, borders=(0.0, 0.5), workers=1, force=False):
        """the packed area masks of an existing BEV Fusion lane store (lane_raster.build_mask_cache) -> its dict + text"""
        from . import lane_gt, lane_raster
        from .suite import DATA
        store = os.path.join(os.path.expanduser(out) if out else DATA, "lane", lane_gt.signature(rig="bev_fusion"))
        if not os.path.isdir(store):
            raise ValueError(f"no BEV Fusion lane store {store}: make it with `generate-bev` or `generate-lane --rig bev_fusion`")
        r = lane_raster.build_mask_cache(store, borders=tuple(borders), force=force, workers=workers)
        return {**r, "text": f"{r['frames']} frames, {len(r['made'])} file(s) made, {len(r['files']) - len(r['made'])} already complete: {os.path.join(store, 'masks')}"}

    @staticmethod
    def build(dataroot="~/data/nuscenes", version="v1.0-mini", out=None, scenes=None, stride=1, workers=1, res=0.5, use_flow=True, use_radar=True, boxes_at="camera"):
        """the low-level writer behind `report` (one full devkit, `out` required; pipeline.build) -> {"written": n}"""
        from . import pipeline
        if not out:
            raise ValueError("build needs out (the folder to write)")
        try:
            return {"written": pipeline.build(os.path.expanduser(dataroot), version, os.path.expanduser(out), scenes, stride, workers, res, **_options(use_flow, use_radar, boxes_at))}
        except SystemExit as e:                                          # a folder of another resolution: build exits with the message, the interface makes it a wrong argument
            raise ValueError(str(e)) from None

    # ------------------------------------------------------------------------------------------------------------ look at GT
    @staticmethod
    def gt_key(*args, **kwargs):
        """the key of what is stored for a system, nothing is generated -> GtKey (gt_api.gt_key); key.complete, key.load(token), key.lane(token), key.mask(token)"""
        from . import gt_api
        return gt_api.gt_key(*args, **kwargs)

    @staticmethod
    def have(set_name=None, scenes=None, version=None, dataroot=None, stride=1, data=None):
        """is the stored front GT of the current generator there for these sequences (the check before a training; sets.stored_for) -> {ok, have, total, missing, text, ...}"""
        from . import current_version, sets
        want = sets.scenes(set_name) if set_name else scenes
        if not want:
            raise ValueError("give --set, --scenes or --scenes-file")
        r = sets.stored_for(want, data, version, dataroot, stride)
        text = f"{r['have']}/{r['total']} keyframes of {r['scenes']} sequences ({r['version']}) are stored for GT version {current_version()}; missing {len(r['missing'])}"
        return {**r, "ok": not r["missing"], "text": text}

    @staticmethod
    def check_data(dataroot, version=None, set_name=None, scenes=None):
        """are the raw nuScenes files the generator reads on disk (tables, maps, CAM_FRONT, radars; sets.check_raw) -> {ok, missing, maps, text}"""
        from . import sets
        r = sets.check_raw(dataroot, version, set_name, scenes)
        return {**r, "ok": not (r["missing"] or r["maps"]), "text": sets.report_raw(r)}

    @staticmethod
    def sets():
        """the scene sets small / full and how much of each is stored -> {text}"""
        return {"text": _sets_text()}

    @staticmethod
    def versions(update=False):
        """the GT versions, the lane versions and the releases; the current one marked, which is stored. A source whose signature has no row yet gets the rows derived from git (the next number, the text and the
        date of the last commit that touched the hashed files); `update` writes them to versions.csv / releases.csv -> {text, proposal}"""
        from . import lane_gt
        from . import versions as V
        from .nuscenes_gt import SIGNATURE_FILES
        from .suite import code_signature
        prop = V.propose(code_signature(), lane_gt.version_of(lane_gt.code_signature()), V.last_change(SIGNATURE_FILES))
        text = _versions_text()
        if prop["front"] or prop["release"]:
            rows = [f"front {prop['front']['version']} = signature {prop['front']['signature']}: {prop['front']['what_changed']}"] if prop["front"] else []
            rows += [f"release {prop['release']['release']} = front {prop['release']['front']} + lane {prop['release']['lane']}"] if prop["release"] else []
            text += "\nrows this source needs (derived from git): " + "; ".join(rows)
            if update:
                V.write_rows(prop)
                tag = (prop["front"] or {}).get("tag")
                text += "\nadded to GtGenerator/versions.csv / releases.csv: review the rows (same_gt_as stays by hand, with a pinned proof), commit" + (f", then git tag -a {tag} -m '...'" if tag else "")
            else:
                text += "\n(`versions --update` adds them)"
        return {"text": text, "proposal": prop}

    @staticmethod
    def area_mask(token, kind="intersection", border_m=0.0, png=None, out=None):
        """one area mask of a keyframe of the BEV Fusion lane store below the store root `out` (default data/autogt): its number of cells, optionally drawn to `png` (lane_raster.area_mask) -> {cells, shape, text, png}"""
        import numpy as np
        from . import lane_gt, lane_raster
        if not 0.0 <= border_m < 100.0:                                  # also refuses nan
            raise ValueError(f"border_m must be 0 or more (metres), got {border_m}")
        g = lane_gt.load(token, data=os.path.join(os.path.expanduser(out), "lane") if out else lane_gt.DATA, rig="bev_fusion")
        m = lane_raster.area_mask(g, kind, border_m)
        text = f"{kind} mask of {token}, border {border_m:g} m: {int(m.sum())} of {m.size} cells ({m.shape[0]} x {m.shape[1]}, x forward 0 .. 60 m, y left -26 .. 26 m)"
        if png:
            import cv2
            img = np.flip(m, (0, 1)).astype(np.uint8) * 255                                  # x forward = up, y left = left, like the pictures of the GT
            if not cv2.imwrite(png, cv2.resize(img, None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST)):
                raise ValueError(f"could not write {png}")
            text += f"\n-> {png}"
        return {"cells": int(m.sum()), "shape": list(m.shape), "png": png, "text": text}

    # ------------------------------------------------------------------------------------------------------------ check
    @staticmethod
    def audit_lane(system, dataset="v1.0-trainval", dataroot=None, scenes=None, out=None, limit=None, verbose=True):
        """audit a stored lane GT before it is released: every keyframe there, every file loads and holds what the version promises (store_audit.audit_lane)
        -> {ok, problems, stats, lines}"""
        from . import lane_gt, store_audit
        from .suite import DATA, default_root
        if limit is not None and limit < 1:
            raise ValueError(f"limit must be 1 or more, got {limit}")
        store = os.path.join(os.path.expanduser(out) if out else DATA, "lane", lane_gt.signature(rig=system))
        if not os.path.isdir(store):
            raise ValueError(f"no lane store {store}")
        return store_audit.audit_lane(store, system, dataroot or default_root(dataset), dataset, scenes, limit, verbose=verbose)

    @staticmethod
    def check(dataroot="~/data/nuscenes", version="v1.0-mini", scenes=None, stride=1, holdout=False):
        """automatic consistency checks of the GT over keyframes (checks.run); raises when a check fails"""
        from . import checks
        from .store import check_arguments
        check_arguments(stride=stride)
        checks.run(os.path.expanduser(dataroot), version, scenes, stride, holdout)
        return {"ok": True}

    @staticmethod
    def report(dataroot=None, version="v1.0-mini", light=False, out=None, data=None, no_data=False, scenes=None, stride=1, workers=1, samples=10, holdout=False,
               use_flow=True, use_radar=True, boxes_at="camera"):
        """evaluate the GT of all sequences, store it, write ONE PDF report with samples (report.report; made at 0.5 m) -> {out, text}"""
        import datetime
        from . import report
        from .store import check_arguments
        from .suite import DATA, default_root
        check_arguments(stride=stride, workers=workers)
        out = out or os.path.join(DATA, "reports", f"autogt_report_{version}_{datetime.date.today().isoformat()}.pdf")
        light = light or "trainval" in version
        note = "note: the leave-one-radar-out test needs the full devkit: skipped with the light devkit (trainval)" if holdout and light else ""
        report.report(os.path.expanduser(dataroot or default_root(version)), version, out, None if no_data else os.path.expanduser(data or DATA), scenes, stride, holdout, samples,
                      workers, light, **_options(use_flow, use_radar, boxes_at))
        return {"out": out, "text": note}

    @staticmethod
    def digest(out=None, compare=None, stride=2):
        """a digest per frame of nuScenes mini (checks.digests): two generators that make the same GT give the same file; `compare` = [A, B]: exit 1 when any frame differs -> {ok, text}"""
        if compare:
            x, y = (json.load(open(p)) for p in compare)
            bad = [k for k in x["digests"] if x["digests"][k] != y["digests"].get(k)] + [k for k in y["digests"] if k not in x["digests"]]
            return {"ok": not bad, "differ": bad, "text": f"{len(x['digests'])} frames, signatures {x['signature']} and {y['signature']}: {len(bad)} differ {bad[:3]}"}
        if not out:
            raise ValueError("give the file to write, or --compare A B")
        from . import checks
        r = checks.digests("~/data/nuscenes", "v1.0-mini", stride)
        with open(out, "w") as f:
            json.dump(r, f)
        return {"ok": True, "text": f"{len(r['digests'])} digests of signature {r['signature']} -> {out}"}

    @staticmethod
    def selftest(only=None):
        """all tests of the suite (5 - 10 min), or only some parts (`only`: names of SELFTESTS) -> {text}"""
        if not only:
            from . import selftest
            selftest.run()
            return {"ok": True}
        import importlib
        import time
        lines = []
        for name in dict.fromkeys(only):                                  # no duplicates, the order asked for
            if name not in SELFTESTS:
                raise ValueError(f"unknown test part {name!r}: {', '.join(sorted(SELFTESTS))}")
            t0 = time.time()
            mod = importlib.import_module(f".{SELFTESTS[name]}", __package__)
            fn = getattr(mod, "run", None) or mod.selftest
            msg = fn(verbose=False) if "verbose" in fn.__code__.co_varnames[:fn.__code__.co_argcount] else fn()
            lines.append(f"{name}: ok ({time.time() - t0:.0f} s)" + (f" {str(msg)[:150]}" if msg else ""))
        return {"ok": True, "text": "\n".join(lines)}

    # ------------------------------------------------------------------------------------------------------------ pictures
    @staticmethod
    def view(out, picks=None, total=False):
        """sample pictures of keyframes (view.run); picks = ["scene-0061:8" (mini, generated now) or a sample token (stored GT)]"""
        from . import view
        with _bad_pick():
            view.run(out, [x.split(":") for x in picks] if picks else None, total=total)
        return {"out": out}

    @staticmethod
    def sheets(out):
        """overview sheets of the sample pictures (view.sheets)"""
        from . import view
        view.sheets(out)
        return {"out": out}

    @staticmethod
    def masks(out):
        """the two coverage masks (BEV fusion: front; Spurmitte: surround) as one picture (rig_masks.picture) -> {out, text}"""
        from . import rig_masks
        path = rig_masks.picture(out)
        return {"out": path, "text": str(path)}

    @staticmethod
    def lanegraph(out, pick, version="v1.0-mini", dataroot=None):
        """picture of the lane graph of a keyframe: the six cameras and the top view; pick = "scene-0061:8" (the scene's 9th keyframe) -> {out, text}"""
        import cv2
        from . import lane_graph_view as V
        from .lane_gt import LaneGT
        from .light_nusc import light_nuscenes
        from .suite import GtGenerator, default_root
        root = os.path.expanduser(dataroot or default_root(version))
        name, sep, k = pick.partition(":")
        if not sep or not k.isdigit():
            raise ValueError(f"pick: scene-0061:8 (the scene and the number of the keyframe), got {pick!r}")
        with _bad_pick():
            nusc = light_nuscenes(root, version, [name], cameras=V.CAMERAS, radars=False)
            gt = GtGenerator(root, version, nusc=nusc)
            tok = gt.keyframes(name)[int(k)]
            g = LaneGT(root, version, nusc=nusc).graph(tok)
        if not cv2.imwrite(out, V.mosaic(gt, tok, g, title=f"{name}:{k} {g['location']}  " + V.legend_text(g))):
            raise ValueError(f"could not write {out}")
        return {"out": out, "text": f"{out}: {V.legend_text(g)}"}

    @staticmethod
    def video(scene, mp4, seconds=0.0, version="v1.0-mini", dataroot=None):
        """drive through the GT of one sequence at 12 Hz (H.264, read back; video.run) -> video.run's dict"""
        from . import video
        from .suite import default_root
        with _bad_pick():
            return video.run(scene, mp4, seconds, dataroot=dataroot or default_root(version), version=version)

    # ------------------------------------------------------------------------------------------------------------ tidy
    @staticmethod
    def clean(delete=False, move=False, videos=False, pictures=False, reports=False, old_outputs=False, stale_caches=None, all_caches=None, older_than=None, include_released=False):
        """list / move generated videos, pictures, reports, old outputs and caches; `delete` only for what one command rebuilds (clean.clean)"""
        from . import clean
        return clean.clean(delete=delete, move=move, videos=videos, pictures=pictures, reports=reports, old_outputs=old_outputs,
                           stale_caches=stale_caches, all_caches=all_caches, older_than=older_than, include_released=include_released)
