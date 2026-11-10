"""scratch: runs tests/test_student_encoder.py as on a clean checkout (the gitignored EfficientViT-B1 weight file 'missing') by pointing the module constant at a path that does not exist; nothing on disk is touched"""
import runpy
import sys

sys.path[:0] = ["training"]
import student_encoder as S  # noqa: E402

S.WEIGHTS["efficientvit_b1"] = "models/pretrained/imagenet/no_such_folder/model.safetensors"
sys.argv = ["tests/test_student_encoder.py"]
runpy.run_path("tests/test_student_encoder.py", run_name="__main__")
