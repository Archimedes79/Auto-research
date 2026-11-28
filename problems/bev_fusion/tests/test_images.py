"""Tests of net/prep/images.py (its former self-test).

    python tests/test_images.py      (from the problem folder, with its module path)
"""
import torch

from net.prep.images import BEVFUSION_MEAN, BEVFUSION_STD, IMG_MEAN, IMG_STD, bevfusion_ida, load_image, nuscenes_front_ida


def _selftest():
    """the two normalisations on a black image (value 0): known numbers by hand; they must never share their constants"""
    import tempfile

    from PIL import Image
    with tempfile.TemporaryDirectory() as tmp:
        p = tmp + "/black.png"
        Image.new("RGB", (1600, 900)).save(p)
        x = load_image(p)
    assert x.shape == (3, 256, 704), x.shape
    want = torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225])
    assert torch.allclose(x[:, 0, 0], want, atol=1e-6), x[:, 0, 0]
    assert IMG_MEAN[0] > 1 > BEVFUSION_MEAN[0] and IMG_STD[0] > 1 > BEVFUSION_STD[0]
    assert torch.equal(bevfusion_ida()[2], torch.tensor([0.0, 0.0, 1.0])) and nuscenes_front_ida().shape == (3, 3)
    print("images selftest ok: the BEVFusion loader normalises a black pixel to (-2.118, -2.036, -1.804); the DualBEV constants stay on 0..255")


if __name__ == "__main__":
    _selftest()
