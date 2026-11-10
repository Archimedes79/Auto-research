"""scratch: shorten the longest lines of the memory index MEMORY.md (one line per entry kept; the detail lives in the topic files). Replaces by the link target, asserts each target once."""
import re

P = "/mnt/c/Users/mmeut/.claude/projects/C--Auto-research/memory/MEMORY.md"
NEW = {
    "bev-fusion-t22-student-encoder.md": "- [t22 student encoder (2026-11-09)](bev-fusion-t22-student-encoder.md) — EfficientViT-B1 stride 8: stage B INCONCLUSIVE, gate C0 INCONCLUSIVE, B2 CUT; R arm + seed twin pre-registered in the card (2729a18bec) but POSTPONED by D68; B3 / stop wait for Mirko; L4 check read (no lane gain)",
    "lane-junction-smear-diagnosis-2026-11-08.md": "- [Junction smear diagnosis (2026-11-08)](lane-junction-smear-diagnosis-2026-11-08.md) - E279 centre map smears junctions: entry->exit pairs unknown (oracle 0.88 vs all pairs 0.53 vs p 0.71); E283 card cf9112fa: offset 1-2.5 m is the largest loss, CPU levers closed, next = CPU scorer oracle",
    "autogt-move-plan.md": "- [autogt status 2026-11-08](autogt-move-plan.md) — l5 (BEV Fusion rig of the lane GT, store db288acca5) merged 181f72fe, tags autogt/lane-l5 + autogt/v12 local; move of data/autogt to D: done later by the GT chat (e6a9b265); front GT v9 untouched",
    "bev-fusion-net-bundle-separation.md": "- [Network as a closed bundle](bev-fusion-net-bundle-separation.md) — Mirko's wish: model.py + losses.py separate from loop / loaders; leaks are selftest / CLI / helper imports (only precompute.network_inputs is left); queued for the Architecture chat",
    "speed-levers-d53-2026-11-04.md": "- [Speed levers D53 (2026-11-04)](speed-levers-d53-2026-11-04.md) — share_mover / splat_fused / sync_free default-off (53e5a22), -5..-9 % step and -1.2 GB (d7fc669); the big stalls needed frozen files: D54, no gain by itself",
    "lane-e276-gt-l4-2026-11-07.md": "- [Lane night E275-E279 (2026-11-08)](lane-e276-gt-l4-2026-11-07.md) — E275 KEEP hard close-node step, E276 TIE, E277 soft step CUT, E279 1.25 m KEEP; long links are the weak spot; WSL page-cache ENOMEM fix (reclaim.py)",
    "review-pass-6-push-2026-11-07.md": "- [Review pass 6 + push (2026-11-07)](review-pass-6-push-2026-11-07.md) — review the other chats, fix, clean, push (f28592b2 + 38 tags one by one, never --tags); defects found, clean-checkout method (worktree + WSL gitdir)",
    "bev-fusion-lane-gt-ceiling-2026-11-08.md": "- [BEV lane GT ceiling + overview page (2026-11-08)](bev-fusion-lane-gt-ceiling-2026-11-08.md) - label-offset ceiling ~0.91 does not explain lane IoU 0.43; hits on paint 41-46 %; val loss still falls at 6000/7000 steps; overview artifact URL",
    "feedback_check-metric-direction.md": "- [Check a metric's direction](feedback_check-metric-direction.md) — bev_fusion metrics.json keys sem.*_iou / bce / objhead / motion / group are LOSSES (lower better); I misread one as a score; read the code and compare dt 0 first",
    "host-copies-d54-2026-11-04.md": "- [Host copies D54 (2026-11-04)](host-copies-d54-2026-11-04.md) — batch[\"host\"], one-copy ego tensors, keyed lift: waits 1122 -> 108/step, step time unchanged; makes pin_memory usable (3.51 -> 3.3 s); e513eec pushed",
    "d51-t21-one-frame-temporal-2026-11-04.md": "- [D51 t21 one-frame temporal (2026-11-04)](d51-t21-one-frame-temporal-2026-11-04.md) — Mirko's overnight order: pair + recurrent temporal fusion with one buffered frame; t20 result (lane lever missed), queue/waiter, confounds",
    "d48-restructure-t20-order-2026-11-03.md": "- [D48 restructure + t20 order (2026-11-03)](d48-restructure-t20-order-2026-11-03.md) — Mirko's ten answers (redesign first, 1 m head into the grid head, CaR lane head untouched, L4 on dual), frozen-frame guards",
    "bev-fusion-scratch-hygiene.md": "- [bev_fusion scratch hygiene](bev-fusion-scratch-hygiene.md) — temp/ 18 GB + C: full (2026-11-01): 13 GB moved to D:, slim_at_end guards, 34 old runs slimmed; old-run slim form must keep opt (empty) + rng",
    "feedback_maintain-reproduction-data.md": "- [Maintain reproduction data](feedback_maintain-reproduction-data.md) — Mirko 2026-11-01: keep recipes in git, result packages name GT version / evaluator / frames / model; ask before removing what explains how a number was made",
    "bev-fusion-d38-footprint-study.md": "- [D38 footprint-share study](bev-fusion-d38-footprint-study.md) — D36-D38 training / evaluation discussion handed to the BEV system chat; study record 09fac68 (centre cell = f>=0.5 for 98 %, occlusion big), proposal (0.1 / 0.5)",
    "bev-fusion-night-2026-11-08-autonomous.md": "- [Autonomous night 2026-11-08](bev-fusion-night-2026-11-08-autonomous.md) - Mirko away: chains queued (chain7, B2, L4 check), t24 lane lift P prepared, intersection mask blocked (frozen files / GT chat)",
    "data-structure-d42-2026-11-02.md": "- [Data structure D42 (2026-11-02)](data-structure-d42-2026-11-02.md) — NuScenes / _cache / _checkpoints / _zips layout, project links (link_storage.sh), archives compared then deleted, 9p small-file pitfall",
    "queue-lock-settle-views-2026-11-04.md": "- [Queue lock/settle/views 2026-11-04](queue-lock-settle-views-2026-11-04.md) — ticket skip under a machine lock, gpu_settle_s + gpu_max_mb gates (start race), queue/status views; committed locally 5409443",
    "training-speedup-queue-2026-11-01.md": "- [Training speedup + queue 2026-11-01](training-speedup-queue-2026-11-01.md) — RAM gate + FIFO queue; step is host-bound (GPU 33 % busy, ~690 syncs/step; bf16 no gain); the rest waited for his word",
    "training-queue-hold-resume-2026-11-03.md": "- [Training queue + hold/resume + speed options (2026-11-03)](training-queue-hold-resume-2026-11-03.md) — trainsuite queue add/serve/hold/resume/drop, run.py --pin, boot-id restart rule, 13 mutants killed",
    "bev-fusion-cards-deleted-d56-2026-11-04.md": "- [Cards deleted D56 (2026-11-04)](bev-fusion-cards-deleted-d56-2026-11-04.md) — Mirko: \"All 12 cards\" deleted (25a6512, tag archive/cards-before-delete-2026-11-04); verdict runs need a new card",
    "feedback_git-bash-ln-copies-and-lx-links.md": "- [Git Bash ln copies, LX links](feedback_git-bash-ln-copies-and-lx-links.md) — `ln -s` on a dir copies it (26.7 GB); list reparse points and remove links alone before any recursive delete; WSL scripts from files via PowerShell",
    "keep3-best-three-and-scratch.md": "- [Best three + scratch clean-up](keep3-best-three-and-scratch.md) — keep3.py / keep3.json (one slot per checkpoint, band, evidence), standing push rule, temp/ ignored + run.py forces scratch into tags",
    "secure-uncommitted-2026-11-03.md": "- [Secure uncommitted 2026-11-03](secure-uncommitted-2026-11-03.md) — 4 own-path commits to main, code in flight parked in refs/backup/wip-2026-11-03, duplicate report folders; what waits for Mirko's yes",
    "bev-lane-loop18-results.md": "- [BEV lane loop 18 results](bev-lane-loop18-results.md) — fork pruning closes the J gap (J 0.265 -> 0.409), map registration + trust, failure-trained route head, numba assembly; pitfalls (run.py / ledger collision)",
}
text = open(P, encoding="utf-8", newline="").read()
nl = "\r\n" if "\r\n" in text else "\n"
lines = text.split(nl)
done = set()
for i, ln in enumerate(lines):
    m = re.match(r"- \[[^\]]*\]\(([^)]+)\)", ln)
    if m and m.group(1) in NEW:
        assert m.group(1) not in done, m.group(1)
        lines[i] = NEW[m.group(1)]
        done.add(m.group(1))
missing = set(NEW) - done
assert not missing, missing
out = nl.join(lines)
open(P, "w", encoding="utf-8", newline="").write(out)
print("replaced", len(done), "lines; bytes", len(text.encode("utf-8")), "->", len(out.encode("utf-8")))
