# Evaluation/render

How the suite draws: the pictures, videos and pages of an evaluation (`../README.md`).

| file | what |
|---|---|
| `figure.py` | the figure engine: a page (title, panels, overlays, legend) -> an image, one style for every view |
| `video.py` | frames -> a video that plays everywhere (H.264 mp4, read back to check; ffmpeg from `$FFMPEG`, then `~/.evaluation_ffmpeg`, then the PATH) |
| `report.py` | a small html writer for the index pages (headings, tables, thumbnails, relative links) |
| `taps.py` | pictures of tapped tensors of a network, their statistics and plausibility flags |
