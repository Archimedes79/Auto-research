# scratch: README edits of the B2 removal (config row, lane_branch row)
EDITS = [
    ("bev_fusion/training/configs/README.md", r"^\| `t22_B2_s0\.json` \|[^\n]*\n", ""),
    ("bev_fusion/training/README.md",
     r"; the OLDER frames of a pair / window for `ModelConfig\.student_frames` 'all' \(t22\):.*?needs its own build\)",
     ""),
]
