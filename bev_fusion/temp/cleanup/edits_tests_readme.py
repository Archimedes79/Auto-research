# scratch: tests/README.md row of test_student_chain.py without section K (stage B2) and without the name student_frames as a switch
EDITS = [
    ("bev_fusion/tests/README.md",
     r"(`test_student_chain\.py` \| the camera student in the chain \(t22 stage B, lane branch included: `ModelConfig\.camera_student` / `student_init` / `student_trainable`) / `student_frames`,",
     r"\1,"),
    ("bev_fusion/tests/README.md",
     r"unknown kind / `student_frames`, `student_frames` 'all' accepted with a student and refused without one,",
     "unknown kind, `student_frames` other than 'current' (the variant 'all' was cut with t22 stage B2),"),
    ("bev_fusion/tests/README.md",
     r" K `student_frames` 'all' \(2026-11-07;.*?in the private copy\); the test removes",
     " The test removes"),
    ("bev_fusion/tests/README.md",
     r"\| passes \(2026-11-05, CPU, 36 s\); every one of 39 planted bugs in a copy of the code is killed",
     "| passes (2026-11-10 after the removal of the B2 code, CPU, 82 s on a shared box); mutation check of 2026-11-05, sections A - J (the scratch scripts `temp/t22/mutate_chain.py`, `mutate_moved.py` are in the tag `bf/pitch-scratch`): every one of 39 planted bugs in a copy of the code is killed"),
]
