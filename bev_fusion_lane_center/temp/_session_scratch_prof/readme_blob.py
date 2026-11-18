import subprocess
def blob(path, pairs, out):
    s = subprocess.run(["git", "show", "HEAD:" + path], capture_output=True, text=True, encoding="utf-8", check=True).stdout
    for a, b in pairs:
        assert s.count(a) == 1, (path, a[:50], s.count(a))
        s = s.replace(a, b)
    open(out, "w", encoding="utf-8", newline="\n").write(s)
blob("bev_fusion_lane_center/training/README.md",
     [("refused with the GPU loader: `--token-drop`, `--teacher rule`, `--distill-from`", "refused with the GPU loader: `--distill-from` (use `--cpu-loader`)")],
     "bev_fusion_lane_center/temp/_blob_train_readme.txt")
blob("bev_fusion_lane_center/training/grid/README.md",
     [("plain head only: prior / blocks / live inputs run the parent's forward", "plain head only: live inputs run the parent's forward"),
      ("with none_weight / pos_loss max / none_weight_near / yskip / the E379 KD term;", "with none_weight / pos_loss max / the E379 KD term;")],
     "bev_fusion_lane_center/temp/_blob_grid_readme.txt")
