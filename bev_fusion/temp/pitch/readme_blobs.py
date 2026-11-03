"""inside commit_cleanup.sh (GIT_INDEX_FILE is set): README blobs = HEAD's text + only my edit, put into the temporary index"""
import subprocess

EDITS = {
    "bev_fusion/tests/README.md": [("(queued for a free GPU: `run_when_gpu_free.sh`)", "(was queued for a free GPU by a waiter script; today `run.py --gpu` does that)")],
    "autoresearch/README.md": [("`project.py` (finds the problem and its `project.json`), `selftest.py` |",
                                "`run_report.py` (one page per training run for a person without code: state, resumes, the card's prediction and rule, curves, the numbers next to reference runs, "
                                "the compute in TOPS, pictures and videos each with a review field that exports a `review.json`; `--ingest` files it; `--selftest`; BEV: `bev_suite.py RUN report`), "
                                "`project.py` (finds the problem and its `project.json`), `selftest.py` |")],
    "bev_fusion/evaluation/README.md": [("python evaluation/bev_suite.py runs/t1_r4_soa quant run --n-scenes 20 ",
                                         "python evaluation/bev_suite.py runs/t1_r4_soa report --vs runs/t8_ctl_s0                # ONE PAGE for a person without code: runs/t1_r4_soa/report/index.html "
                                         "(state, prediction / rule, curves, numbers next to the control, TOPS, 6 random frames + a video, review fields; made first when missing)\n"
                                         "python evaluation/bev_suite.py runs/t1_r4_soa quant run --n-scenes 20 ")],
}
for path, edits in EDITS.items():
    text = subprocess.run(["git", "show", "HEAD:" + path], capture_output=True, text=True, check=True).stdout
    for a, b in edits:
        assert text.count(a) == 1, (path, a[:40], text.count(a))
        text = text.replace(a, b)
    blob = subprocess.run(["git", "hash-object", "-w", "--stdin"], input=text, capture_output=True, text=True, check=True).stdout.strip()
    subprocess.run(["git", "update-index", "--cacheinfo", f"100644,{blob},{path}"], check=True)
    print("readme blob", path)
