for path, pairs in (("bev_fusion_lane_center/training/README.md", [("refused with the GPU loader: `--token-drop`, `--teacher rule`, `--distill-from`", "refused with the GPU loader: `--distill-from` (use `--cpu-loader`)")]),
                    ("bev_fusion_lane_center/training/grid/README.md", [("plain head only: prior / blocks / live inputs run the parent's forward", "plain head only: live inputs run the parent's forward"), ("with none_weight / pos_loss max / none_weight_near / yskip / the E379 KD term;", "with none_weight / pos_loss max / the E379 KD term;")])):
    s = open(path, encoding="utf-8").read()
    for a, b in pairs:
        print(path[-20:], s.count(a))
        s = s.replace(a, b)
    open(path, "w", encoding="utf-8", newline="\n").write(s)
