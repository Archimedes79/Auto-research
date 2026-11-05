# E273.report: the report of experiment E273 (pictures and videos)

`index.html` is the report (open it from this folder: images and videos are relative links, fonts come from Google Fonts when online). The same page is published as a private artifact (https://claude.ai/artifact/Vdey4Vj4D6CD942ymy1qwB). `page.html` is the body for the artifact. `img/` holds JPEG copies of the figures, `vid/` the scene videos as WebM (VP9, 1280 px wide; the original 1920 x 1080 MP4 files are in `../E273.p_s0/videos/` and `../E273.p_s1/videos/`). Binaries are not in git; only this README is.

Every number comes from the scoring cards `runs/e273/<run>/card.json`, the timing files, the gate logs of `runs/e273_p_s*_p7/log.txt` and `runs/_superseded/`, and per-frame scores and a node-drop probe on the oracle graph made with the same scorer for the report. Verdict and rule: card `experiments/E273_*/experiment.md` section 10; index `../E273/README.md`.

Rebuild (scripts are scratch, kept in the git tag `lcr/e273-scratch-report`; check them out into `temp/e273/`): `report_data.py` (per-frame metrics of both seeds, situation breakdown, node-drop probe; GPU about 15 min), `render_frames.py` (best / median / worst frames), `build_report.py` (charts, WebM transcoding, page; template `report_template.html`, charts `svgcharts.py`). Run each with `wsl bash tools/env/_py14.sh <script>`.
