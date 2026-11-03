"""the narrative of the t16 report (scratch): TITLE, LEAD, BLOCKS = [(heading, html)]; numbers are read from the stored evaluation tables of the runs where they stand in a sentence"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def T(run, n=1000):
    d = {}
    p = f"bev_fusion/runs/{run}/eval/by_n/table_n{n}.txt"
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            m = re.match(r"^(\S+)\s+(-?[0-9.]+(?:e-?[0-9]+)?|nan)\s*$", line.rstrip())
            if m:
                d[m.group(1)] = float(m.group(2))
    return d


B = T("t16_full_s0")


def g(key, fmt="{:.3f}", t=B):
    return fmt.format(t[key]) if key in t else "n/a"


def svg(path):
    return open(path, encoding="utf-8").read() if os.path.exists(path) else ""


TITLE = "BEV system report: the full network (camera + radar + temporal fusion + motion block) on ground truth v8"
LEAD = ("Written for Mirko (order D44 / D46 of 2026-11-02): generate a full network with fusion, train it on a small set, time it, probe every module, fix, retrain, train on the large set, "
        "establish a baseline, report with videos and images, then iterate on the modules. This page is the report; the working record is the card "
        "<code>bev_fusion/experiments/t16_full_network/experiment.md</code> and the ledger <code>bev_fusion/records/LEDGER.md</code>.")

SUMMARY = f"""
<div class=card><b>In one paragraph.</b> The five-layer framework (precompute, input interface, network slots, output interface, post-processing) carries a full network: dual camera features, delta-slot radar encoder,
camera memory, concat fusion, DualBEV trunk, map / object / velocity heads with the velocity hurdle, and the motion block in its joint form. It was brought up on a 10-scene small set (smoke test, 400-step runs, module probes, timing,
8 findings, 2 fixes), then trained for 6000 steps on 428 scenes. On 1000 validation frames (single seed, GT v8) the baseline reaches drivable-area IoU {g('seg_fov/seg/dt+0.00/drivable_area/all/all/iou@0')},
lane-divider IoU {g('seg_fov/seg/dt+0.00/lane_divider/all/all/iou@0.5')}, vehicle AP@2 m {g('detf/dt+0.00/vehicle/all/ap@2')}, and the velocity field is {g('vel/dt+0.00/centre/field/moving/epe')} m/s off at moving
box centres (the zero predictor: {g('vel/dt+0.00/centre/zero/moving/epe')}).</div>
<ul>
<li><b>It works end to end:</b> training, evaluation, module probes, timing, videos and the runner all run through the tested framework; every gate passes (<code>bash tests/run_gates.sh</code>); the work is on GitHub (<code>main</code>).</li>
<li><b>The motion block is the new capability:</b> a velocity field that responds to the radar Doppler. After 6000 steps its error on static cells is {g('vel/dt+0.00/cells/static/epe')} m/s (it was 1.26 m/s after 400 steps: early-training noise, as the Temporal Fusion chat expected).</li>
<li><b>Weak spots:</b> thin and rare map classes (lane divider, road divider, pedestrian crossing, car park) and pedestrians / two-wheelers are far below the vehicles and the drivable area; the train-to-val gap of the six map IoUs is about 0.09, and 64 % more training scenes at the same step budget (arm t17) did not close it (gap +2 %, no map gain: see Results).</li>
<li><b>Cost:</b> the new blocks add about 12 % to the inference time of the cached chain (96.4 ms against 86.2 ms per frame at batch 1 on an idle RTX 3060, 3 time offsets, small-set runs) and 3.6 s per training step at accumulation 2.</li>
<li><b>Honest limits:</b> every comparison is single seed; the control, the data arm and the bundle arm are labelled "by hand: one seed"; the GT version changed (v5 to v8), so numbers of older runs are not comparable.</li>
</ul>
"""

SYSTEM = f"""
<p>The system is drawn from the code (<code>bev_fusion/training/architecture.svg</code>, file map in <code>training/architecture.md</code>). A camera path and a radar path are brought to one BEV grid (120 x 104 cells of 0.5 m, x 0 - 60 m, y +-26 m),
fused, passed through a trunk and decoded by heads at any time offset dt (-0.5 / 0 / +0.5 s). Everything on the training path is dense and differentiable; box selection and line thresholding stay in post-processing.</p>
<table><tr><th>slot</th><th>what the full network uses</th></tr>
<tr><td>camera encoder</td><td>two frozen image encoders with cached outputs (DualBEV heads, stride 16; BEVFusion map-trained Swin-T, stride 8), two depth-weighted lifts merged by a 1x1 conv, 3 frames</td></tr>
<tr><td>camera memory</td><td>window of the last frames + a world-aligned static memory</td></tr>
<tr><td>radar encoder</td><td>delta-coded slot raster (3 exact slots per cell + remainder), lossless side loss that decays over the run; radar detections of 0.5 s compensated to now</td></tr>
<tr><td>fusion, trunk</td><td>concat fusion with modality dropout; DualBEV trunk (pre-trained, width 1.0) with a detail path</td></tr>
<tr><td>heads</td><td>5 area layers, 2 line layers + distance fields, dense object maps; object head with sign-free direction and the moving / static hurdle; radar auxiliary head</td></tr>
<tr><td>motion block</td><td>velocity field from two camera frames + the radar Doppler in a 3-level pyramid; the object head's box velocity is read from it; its gradient is clipped on its own</td></tr>
<tr><td>left out</td><td>radar memory with the Doppler static gate (an unlearned block in the small-set probe), lane branch, box refiner, centre-line head (no head design yet)</td></tr></table>
<div style="overflow:auto">{svg('bev_fusion/training/architecture.svg')}</div>
"""

BRINGUP = """
<p>All small-set runs went through the launcher as diagnostics (no verdict). Gates and findings, in order:</p>
<ul>
<li><b>S0 CPU gates:</b> the slot test builds the full combination and sends a gradient to every trainable parameter; the gate run found one stale assertion (velocity-hurdle test), fixed.</li>
<li><b>S1 smoke (20 steps):</b> finite loss, a gradient on all 14 modules, peak GPU memory 10.1 GB.</li>
<li><b>S2 small run (400 steps, 8 scenes):</b> no alert; validation loss 20.2 to 18.2.</li>
<li><b>S4 module probe:</b> every module's output is finite, not constant, without dead channels, and responds when its input is removed (without the camera the camera BEV changes by 0.97, without all radar detections the radar raster by 1.0 and the velocity field by 0.24 in the 400-step run).</li>
</ul>
<table><tr><th>#</th><th>finding</th><th>what was done</th></tr>
<tr><td>1</td><td>the motion block's gradient norm (29 - 133) dominates the global clip (the others reach 41)</td><td>opt-in separate clip for that block (<code>grad_clip_split</code>, tested, critic pass), used in the large run</td></tr>
<tr><td>2</td><td>static-cell velocity error 1.26 m/s after 400 steps (control 0.51)</td><td>zero-mean noise, not an ego-motion offset; a 10 x background weight changed it little; it settled to 0.19 m/s in the 6000-step run</td></tr>
<tr><td>3</td><td>the radar memory's static map is exactly 1.0 everywhere, its velocity head gets no gradient</td><td>off in the large run; a Doppler-prior card (t19) exists for later</td></tr>
<tr><td>4</td><td>the radar-kept monitor fell 0.94 to 0.04 in 400 steps</td><td>expected for a short schedule; it is 0.90 at the end of the 6000-step run</td></tr>
<tr><td>5 - 8</td><td>the hurdle's moving probability is high away from objects (unsupervised cells); the debug taps did not cover the new modules; rare map classes still at IoU 0 after 400 steps; the v8 GT cache costs 0.57 s per sample and worker when cold</td><td>taps added by the Evaluation chat; no prefill needed (6 loader workers keep up)</td></tr></table>
<p>A matched small-set comparison (400 steps, 8 scenes, one seed each) showed the full network better on moving objects (velocity error at box centres 1.40 against 1.92 m/s) and worse on static cells; it is a diagnostic, not a verdict.</p>
"""

RESULTS = """
<p>VAL, 1000 frames spread over the 82 validation scenes, cold memory per frame, GT v8 (<code>c643a7539d</code>), scoring on the fusion area (cells that camera and radar both observe, D37).
The columns after the baseline appear when those runs have finished; the small green / red number is the difference to the baseline, one seed per run.</p>
"""

COST = f"""
<ul>
<li><b>Inference</b> (batch 1, three time offsets, cached chain without the frozen image heads, idle RTX 3060): full network 96.4 ms, control chain 86.2 ms (small-set runs, step 400); per module: lift 14.0, camera memory 13.9, trunk 10.9, heads 10.0 (3 calls), motion block 4.5, radar encoder 4.0 ms. An idle-GPU timing of the large runs is added to the final version of this report when the GPU is free.</li>
<li><b>Training</b> on the large set: 3.61 s per optimiser step (18 keyframes), 7 % waiting for the data once the GT cache is warm; peak GPU memory 10.1 GB, host RAM about 10 GB; 6000 steps take 6 h.</li>
<li><b>Training-code speed work</b> (Training chat): the step is host bound (GPU kernels busy about a third of the time); fp16 with a gradient scaler is built and tested, its speed is not measured yet.</li>
</ul>
"""

WEAK = """
<ul>
<li>Thin map classes (lane and road dividers) and rare classes are the weakest outputs; the baseline's train-to-val gap is large for them (carpark 0.17, lane divider 0.13, road divider 0.10); the data arm t17 (700 instead of 428 scenes) did not reduce the gap and gave no map gain, so the next levers are the features and the training schedule, not the amount of data alone.</li>
<li>Pedestrians and two-wheelers: AP@2 m far below vehicles.</li>
<li>The hurdle's moving probability is not calibrated away from objects; the dense velocity field is only supervised near objects and on a weak background term.</li>
<li>Open findings of the bug hunt (<code>records/open_findings_2026-11-02.md</code>): w_unknown / w_interpolated were ignored by the object head, unknown-velocity objects trained the moving / static head toward static (0.7 % of the observed boxes), the batch decoder accepts dts it cannot decode; a bundle arm (t18) tests the fixes together with the motion block's own learning rate.</li>
<li>Single seeds everywhere: the seed spread of this recipe is not measured; the first open item is a seed twin of the baseline.</li>
</ul>
"""

SOA = """
<p>The older stored comparison (experiment t6, GT v5, camera cells, 3000-step chain without the new blocks) put the chain at 17 - 85 % of the published GaussianCaR model's IoU per map layer (drivable 0.666 against 0.845, walkway 0.191 against 0.594, lane divider 0.181 against 0.691, scored on different subsets).
The new baseline is scored differently (GT v8, fusion area), so its map numbers are higher but not comparable with those; the published model was not re-scored on v8. The matched comparison available today is the control run (the old chain's recipe on the same data and frames).
Where the new network clearly adds something that the old chain did not have is the velocity: a dense field that uses the radar Doppler.</p>
"""

HOWTO = """
<p>From <code>bev_fusion/</code> in the WSL (env <code>gcar</code>); the full table is in <code>bev_fusion/README.md</code>.</p>
<ul>
<li>tests: <code>bash tests/run_gates.sh</code></li>
<li>train: from <code>autoresearch/</code>: <code>AR_PROJECT=../bev_fusion python3 -m trainsuite start ../bev_fusion/training/configs/t16_full.json --id &lt;id&gt; --hyp "..." --pred "..." --rule "..." --go</code></li>
<li>evaluate: <code>python evaluation/bev_suite.py runs/&lt;run&gt; --curves --fusion --velocity eval --n 1000 --prior 500</code>; then <code>modules</code>, <code>timing</code>, <code>qual videos 3</code>, <code>generalisation --n 1000</code>, <code>report --vs runs/&lt;control&gt;</code></li>
<li>the trained systems: <code>runs/t16_full_s0</code> (baseline), further runs as listed in the ledger; the checkpoints stay complete (best three rule)</li>
</ul>
"""

FILES = """
<ul>
<li>cards: <code>bev_fusion/experiments/t16_full_network/</code> (bring-up log), <code>t17_more_data/</code>, <code>t18_motion_lr/</code>, <code>t19_radar_memory/</code> (draft)</li>
<li>decisions: <code>bev_fusion_lane_center/experiment_execution/DECISIONS.md</code> (D44 order, D46 deadline / mail / shutdown)</li>
<li>open lists: <code>bev_fusion/records/open_findings_2026-11-02.md</code>, <code>bev_fusion/records/cleanup_inventory_2026-11-02.md</code></li>
<li>this report and the videos: <code>bev_fusion/results/t16/</code> (not in git, binary)</li>
</ul>
"""

BLOCKS = [("Summary", SUMMARY), ("The system", SYSTEM), ("Bring-up on the small set", BRINGUP), ("Results on the validation set", RESULTS),
          ("Cost and timing", COST), ("Weak spots", WEAK), ("Position against the state of the art", SOA), ("How to run it", HOWTO), ("Where things are", FILES)]
