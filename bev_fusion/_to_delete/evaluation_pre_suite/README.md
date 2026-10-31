# evaluation_pre_suite - the evaluation scripts of before the evaluation suite (moved 2026-10-30)

Replaced by `autoresearch/evalsuite/` (core) + `evaluation/bev_adapter.py`, `bev_task.py`, `bev_views.py`, `bev_suite.py`, `bev_probe.py`
(Mirko's decision: replace the old scripts and fix all callers). Kept here because nothing is deleted; git history has them as well.

| file | what it was | now |
|---|---|---|
| `evaluate_run.py` | the evaluation of a run (metrics.json, table, images, latency, paired bootstrap), plus a namespace of chain helpers that diagnostic scripts imported | `bev_suite.py RUN eval / time / compare`; helpers in `bev_probe.py`; a 40-line forwarding stub stays at `evaluation/evaluate_run.py` while `temp/study/lane3.sh` runs (it calls the old command line) |
| `observe_run.py` | the observability report (stage pictures, checks, index.html, scene video) | `bev_suite.py RUN qual random / frame / debug / videos` (views `overview`, `outputs`, `stages`, `dashboard`) |
| `eval_memory.py` | warm vs cold memory on the last keyframe of val clips | `bev_suite.py RUN memory` (state `warmup` vs `reset`, same frames, paired bootstrap) |
| `video_encode.py` | frames -> H.264 mp4 with read-back check | `autoresearch/evalsuite/render/video.py` |
| `patch_boot.py`, `patch_eval.py` | one-off patch scripts that edited `evaluate_run.py` in place (study lane) | obsolete |
| `evaluate_run_stub.py` | the 40-line forwarding stub that was `evaluation/evaluate_run.py` (old command line -> `bev_suite.py`) | moved 2026-10-31 (repository review cleanup, D24): no caller left (lane 3 ended; `temp/study/lane4.sh` and `temp/t7/finish2.sh` call `bev_suite.py`); its check in `evaluation/bev_selftest.py` was removed |
