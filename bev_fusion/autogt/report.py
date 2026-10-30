"""One-command GT evaluation of all sequences: python -m autogt report --version v1.0-mini --out REPORT.pdf [--data DIR] [--workers N]

For every keyframe of every sequence the generator runs ONCE; the result is stored in the cache (`DIR/<signature>/<token>.npz`, the
same format as `build`, existing files are kept), fed to the automatic checks, and - for a few evenly spread sequences - drawn
as a sample. The report is one PDF: summary of all checks, a table per sequence, optionally the leave-one-radar-out radar test,
the legend and the sample pictures. The samples are drawn straight into the PDF: no picture file is ever written, so there is
nothing to clean up afterwards except the report itself and the stored data, both covered by `python -m autogt clean`
(`--reports`, `--stale-caches`, `--all-caches`).

Big datasets (v1.0-trainval, 850 sequences, 34,149 keyframes):
- `--light` (automatic for trainval): the memory-light devkit (autogt/light_nusc.py); the sequences are processed in groups of
  `GROUP` (60), each group loads only its own records: about 1 GB instead of 8+ GB for the full devkit
- `--workers N`: N processes (fork: they share the devkit loaded once); the unit of work is one sequence. A finished sequence is saved
  as `DIR/<signature>/_scene_<name>.pkl` (its check accumulator and sample): an interrupted run (Ctrl-C, WSL restart, a stopped job)
  resumes with the next sequence, finished ones are not recomputed; a sequence that fails (missing file) is listed in the report
  and skipped
"""
import datetime
import gc
import multiprocessing as mp
import os
import pickle
import textwrap
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from .checks import SRC_HASH, Checks, leave_one_radar_out, radar_fov  # noqa: E402
from .cli import _flat, write_index  # noqa: E402
from .light_nusc import scene_names  # noqa: E402
from .suite import AutoGT  # noqa: E402
from .view import LEGEND, figure  # noqa: E402

_STATE = {}
PAGE_LINES = 78
GROUP = 60


def _text_pages(pdf, title, lines, size=7.5):
    wrapped = []
    for ln in lines:
        wrapped.extend(textwrap.wrap(ln, 190, subsequent_indent="    ") or [""])
    chunks = [wrapped[i:i + PAGE_LINES] for i in range(0, max(len(wrapped), 1), PAGE_LINES)]
    for ci, chunk in enumerate(chunks):
        fig = plt.figure(figsize=(16.5, 11.7))
        fig.text(0.03, 0.97, title + (f" (page {ci + 1} of {len(chunks)})" if len(chunks) > 1 else ""), fontsize=14, va="top", weight="bold")
        fig.text(0.03, 0.93, "\n".join(chunk), fontsize=size, va="top", family="monospace")
        pdf.savefig(fig)
        plt.close(fig)


def _scene_task(name):
    """one sequence: generate every keyframe once, store it, feed the checks; resumable through a pickle per sequence.
    Returns (name, Checks or None, (k, gt) sample or None, error or None)."""
    gen, cfg = _STATE["gen"], _STATE["cfg"]
    sig_dir, stride, want_sample = cfg["sig_dir"], cfg["stride"], name in cfg["pick"]
    pk = os.path.join(sig_dir, f"_scene_{name}.pkl") if sig_dir else None
    if pk and os.path.exists(pk):
        try:
            out = pickle.load(open(pk, "rb"))
            ck_old = out[1]
            # reuse only a pickle that fits THIS run: same stride, a sample exactly where one is wanted, same version of the checks
            # (pickles of before 2026-10-30 carry no version: accepted, their checks.py is the one that has not changed since)
            fits = (out[3] is None and ck_old is not None and ck_old.per_scene[name]["keyframes"] == len(range(0, len(gen.keyframes(name)), stride))
                    and (out[2] is not None) == want_sample and getattr(ck_old, "version", SRC_HASH) == SRC_HASH)
            if fits:
                ck_old.written = 0                                  # nothing was written in THIS run
                return out
        except Exception:                                           # noqa: BLE001  (a half-written file: recompute)
            pass
    try:
        ck, sample, written = Checks(gen.grid), None, 0
        toks = gen.keyframes(name)
        mid = len(toks) // 2
        for k, tok in enumerate(toks):
            if k % stride:
                continue
            gk = gen.keyframe(tok)
            ck.feed(name, k, tok, gk, gen.nusc.get("sample", tok)["timestamp"])
            if sig_dir:
                path = os.path.join(sig_dir, tok + ".npz")
                if not os.path.exists(path):
                    np.savez_compressed(path + ".tmp.npz", **_flat(gk))
                    os.replace(path + ".tmp.npz", path)
                    written += 1
            if want_sample and sample is None and k >= mid:
                sample = (k, gk)
        ck.written = written
        out = (name, ck, sample, None)
        if pk:
            pickle.dump(out, open(pk + ".tmp", "wb"))
            os.replace(pk + ".tmp", pk)
        return out
    except Exception as e:                                          # noqa: BLE001
        return (name, None, None, f"{type(e).__name__}: {e}")


def report(dataroot, version, out_pdf, data_dir=None, scenes=None, stride=1, holdout=False, samples=10, workers=1, light=False,
           verbose=True, **options):
    """see the module docstring; returns dict(pdf, signature, keyframes, written, stored, samples, failed)."""
    if light:
        names = scenes or scene_names(dataroot, version)
        gen0 = AutoGT(dataroot, version, light=True, scenes=names[:1], **options)
    else:
        gen0 = AutoGT(dataroot, version, **options)
        names = scenes or gen0.scenes()
    sig = gen0.signature()
    grid = gen0.grid
    sig_dir = os.path.join(data_dir, sig) if data_dir else None
    if sig_dir:
        os.makedirs(sig_dir, exist_ok=True)
    pick = set(names[i] for i in np.unique(np.linspace(0, len(names) - 1, min(samples, len(names))).round().astype(int))) if samples else set()
    results, t0, done = {}, time.time(), 0
    groups = [names[i:i + GROUP] for i in range(0, len(names), GROUP)] if light else [names]
    for grp in groups:
        gen = AutoGT(dataroot, version, light=True, scenes=grp, **options) if light else gen0
        _STATE["gen"], _STATE["cfg"] = gen, {"sig_dir": sig_dir, "stride": stride, "pick": pick}
        its = None
        if workers > 1:
            pool = mp.get_context("fork").Pool(workers)
            its = pool.imap_unordered(_scene_task, grp)
        else:
            its = (_scene_task(n) for n in grp)
        for r in its:
            results[r[0]] = r
            done += 1
            if verbose:
                print(f"[{done}/{len(names)}] {r[0]} {'FAILED ' + r[3] if r[3] else 'ok'}  {time.time() - t0:.0f} s, "
                      f"ETA {(time.time() - t0) / done * (len(names) - done) / 60:.0f} min", flush=True)
        if workers > 1:
            pool.close()
            pool.join()
        _STATE.clear()
        del gen, its
        gc.collect()
    # the sample pictures and the radar field of view need the devkit of a few sequences only
    keep = sorted(n for n in names if results[n][2] is not None) or names[:1]
    gen = AutoGT(dataroot, version, light=True, scenes=keep, **options) if light else gen0
    ck = Checks(grid)
    failed = {n: r[3] for n, r in results.items() if r[3]}
    held, written = {}, 0
    for n in names:                                                   # merged in sequence order
        _, c, sample, err = results[n]
        if c is not None:
            ck.merge(c)
            written += getattr(c, "written", 0)
            if sample is not None:
                held[n] = sample
    n_kf = len(ck.st["dt_ms"])
    if sig_dir:
        write_index(sig_dir, gen, version, [n for n in names if n not in failed], stride, n_kf, options)
    fov = radar_fov(gen, [n for n in keep if n not in failed], stride)
    ho = leave_one_radar_out(gen, [n for n in names if n not in failed], stride) if (holdout and not light) else None
    os.makedirs(os.path.dirname(os.path.abspath(out_pdf)), exist_ok=True)
    with PdfPages(out_pdf) as pdf:
        head = [f"GT evaluation of {len(names) - len(failed)} sequences, {n_kf} keyframes - nuScenes {version}, grid 120 x 104 at 0.5 m",
                f"generator signature {sig} (options: {options or 'defaults'}), made {datetime.datetime.now():%Y-%m-%d %H:%M}, "
                f"{workers} worker(s), {time.time() - t0:.0f} s",
                f"stored GT: {sig_dir if sig_dir else 'not stored'} ({written} keyframes written in this run, others already there)",
                f"failed sequences: {failed if failed else 'none'}", ""] + ck.text(fov)
        _text_pages(pdf, "autoGT report - summary of the checks", head, size=8)
        rows = ["sequence      keyframes  observed  moving  stationary  ignored  velocity overrides/conflicts  drivable share"]
        rows += [f"{n:12s} {kf:9d} {ob:9d} {mv:7d} {st:11d} {ig:8d} {fl:14d}              {dr:12.0%}"
                 for n, kf, ob, mv, st, ig, fl, dr in ck.scene_rows()]
        if ho:
            rows += ["", "Leave-one-radar-out test of the radar correction:"] + ho
        _text_pages(pdf, "per sequence" + (" and the radar hold-out test" if ho else ""), rows, size=8)
        _text_pages(pdf, "how to read the sample pictures", LEGEND + ["", "ignored = box without any lidar / radar point; stationary = one fixed "
                    "world pose, velocity 0; flags: '!' radar conflict, 'o' radar override (the radar fixed the track's speed)."])
        for name, (k, gk) in held.items():
            fig = figure(gen, name, k, None, gt=gk)
            pdf.savefig(fig)
            plt.close(fig)
    info = {"pdf": os.path.abspath(out_pdf), "signature": sig, "keyframes": n_kf, "written": written, "stored": sig_dir,
            "samples": list(held), "failed": failed}
    if verbose:
        print(f"report: {info['pdf']} ({os.path.getsize(out_pdf) / 1e6:.1f} MB), {len(held)} samples, signature {sig}, "
              f"{len(failed)} failed sequences")
    return info
