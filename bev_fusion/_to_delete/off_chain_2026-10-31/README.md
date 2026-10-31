# off_chain_2026-10-31 - taken out of the live tree by the cleanup of the repository review (Mirko's decision D24)

Review: `../../../autoresearch/history/repo_review_2026-10-31.md` section 7. Names are `<origin folder>__<file>`. Nothing is deleted;
the git history keeps them as well.

| file | what it was | why it is here |
|---|---|---|
| `training__distill_light_encoder.py` | distillation of the frozen DualBEV heads into the light image encoder (plan 2.7) | imported by nothing live; its three runs crashed on 2026-10-30 (`runs/distill_queue.log`: it borrowed `FrontRadarDataset._image`, which now calls `_image_uint8`); restore and fix it if plan 2.7 resumes (the encoder itself is `training/legacy/light_encoder.py`) |
| `timing__run_e8_queue.sh` | one-off queue of 2026-10-30: S1 timing, S1 synthetic gate, the three distillations | finished (`temp/timing/run_e8_queue.log`); not running |
| `tests__run_after_t1.sh` | waited until no `t1_r*` training round ran, then for a free GPU | stale (the t1 rounds ended on 2026-10-27); its only caller was `timing__run_e8_queue.sh`; not running |
