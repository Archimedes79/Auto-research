"""One-command GT evaluation of all sequences: python -m autogt report --version v1.0-mini --out REPORT.pdf [--data DIR]

For every keyframe of every sequence the generator runs ONCE; the result is stored in the cache (`DIR/<signature>/<token>.npz`, the
same format as `build`, existing files are kept), fed to the automatic checks, and - for a few evenly spread sequences - drawn
as a sample. The report is one PDF: summary of all checks, a table per sequence, optionally the leave-one-radar-out radar test,
the legend and the sample pictures. The samples are drawn straight into the PDF: no picture file is ever written, so there is
nothing to clean up afterwards except the report itself and the stored data, both covered by `python -m autogt clean`
(`--reports`, `--stale-caches`, `--all-caches`).
"""
import datetime
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from .checks import Checks, leave_one_radar_out, radar_fov  # noqa: E402
from .cli import _flat, write_index  # noqa: E402
from .suite import AutoGT  # noqa: E402
from .view import LEGEND, figure  # noqa: E402


def _text_page(pdf, title, lines, size=7.5):
    fig = plt.figure(figsize=(16.5, 11.7))
    fig.text(0.03, 0.97, title, fontsize=14, va="top", weight="bold")
    body = []
    for ln in lines:
        body.extend(textwrap.wrap(ln, 190, subsequent_indent="    ") or [""])
    fig.text(0.03, 0.93, "\n".join(body), fontsize=size, va="top", family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


def report(dataroot, version, out_pdf, data_dir=None, scenes=None, stride=1, holdout=False, samples=10, verbose=True, **options):
    """see the module docstring; returns dict(pdf, signature, keyframes, written, stored, samples)."""
    gen = AutoGT(dataroot, version, **options)
    sig = gen.signature()
    names = scenes or gen.scenes()
    sig_dir = os.path.join(data_dir, sig) if data_dir else None
    if sig_dir:
        os.makedirs(sig_dir, exist_ok=True)
    pick = set(names[i] for i in np.unique(np.linspace(0, len(names) - 1, min(samples, len(names))).round().astype(int))) if samples else set()
    ck = Checks(gen)
    held, written, n_kf = {}, 0, 0
    for name in names:
        toks = gen.keyframes(name)
        mid = len(toks) // 2
        for k, tok in enumerate(toks):
            if k % stride:
                continue
            gk = gen.keyframe(tok)
            n_kf += 1
            ck.feed(name, k, tok, gk)
            if sig_dir and not os.path.exists(os.path.join(sig_dir, tok + ".npz")):
                path = os.path.join(sig_dir, tok + ".npz")
                np.savez_compressed(path + ".tmp.npz", **_flat(gk))
                os.replace(path + ".tmp.npz", path)
                written += 1
            if name in pick and k >= mid and name not in held:
                held[name] = (k, gk)                                   # the sample of this sequence (kept in memory only)
        if verbose:
            print(f"{name}: {len(toks)} keyframes", flush=True)
    if sig_dir:
        write_index(sig_dir, gen, version, names, stride, n_kf, options)
    fov = radar_fov(gen, names, stride)
    ho = leave_one_radar_out(gen, names, stride) if holdout else None
    os.makedirs(os.path.dirname(os.path.abspath(out_pdf)), exist_ok=True)
    with PdfPages(out_pdf) as pdf:
        head = [f"GT evaluation of {len(names)} sequences, {n_kf} keyframes - nuScenes {version}, grid 120 x 104 at 0.5 m",
                f"generator signature {sig} (options: {options or 'defaults'}), made {datetime.datetime.now():%Y-%m-%d %H:%M}",
                f"stored GT: {sig_dir if sig_dir else 'not stored'} ({written} keyframes written, {n_kf - written} already there)",
                ""] + ck.text(fov)
        _text_page(pdf, "autoGT report - summary of the checks", head, size=8)
        rows = ["sequence      keyframes  observed  moving  stationary  ignored  velocity overrides/conflicts  drivable share"]
        rows += [f"{n:12s} {kf:9d} {ob:9d} {mv:7d} {st:11d} {ig:8d} {fl:14d}              {dr:12.0%}"
                 for n, kf, ob, mv, st, ig, fl, dr in ck.scene_rows()]
        if ho:
            rows += ["", "Leave-one-radar-out test of the radar correction:"] + ho
        _text_page(pdf, "per sequence" + (" and the radar hold-out test" if ho else ""), rows, size=8)
        _text_page(pdf, "how to read the sample pictures", LEGEND + ["", "ignored = box without any lidar / radar point; stationary = one fixed "
                   "world pose, velocity 0; flags: '!' radar conflict, 'o' radar override (the radar fixed the track's speed)."])
        for name, (k, gk) in held.items():
            fig = figure(gen, name, k, None, gt=gk)
            pdf.savefig(fig)
            plt.close(fig)
    info = {"pdf": os.path.abspath(out_pdf), "signature": sig, "keyframes": n_kf, "written": written, "stored": sig_dir,
            "samples": list(held)}
    if verbose:
        print(f"report: {info['pdf']} ({os.path.getsize(out_pdf) / 1e6:.1f} MB), {len(held)} samples, signature {sig}")
    return info
