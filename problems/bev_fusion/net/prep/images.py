"""What an image goes through before the first learned weight: the normalisation constants, the pixel transforms of the two camera pipelines, the calibration vector
of the DualBEV depth net and the image loader of the BEVFusion pipeline."""
import numpy as np
import torch


IMG_MEAN = (123.675, 116.28, 103.53)                  # RGB, 0..255


IMG_STD = (58.395, 57.12, 57.375)


def camera_vector(K, post_rot, post_tran, sensor2ego, bda=None):
    """DualBEV mlp_input (27 values per camera): fx fy cx cy, post_rot 00 01, post_tran x, post_rot 10 11,
    post_tran y, bda 00 01 10 11 22, sensor2ego 3 x 4 row-wise. Inputs: tensors (B, 3, 3), (B, 2, 2) or (B, 3, 3),
    (B, 2) or (B, 3), (B, 4, 4) camera -> vehicle; bda (B, 3, 3) or identity."""
    B = K.shape[0]
    if bda is None:
        bda = torch.eye(3, dtype=K.dtype, device=K.device).expand(B, 3, 3)
    v = [K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2],
         post_rot[:, 0, 0], post_rot[:, 0, 1], post_tran[:, 0], post_rot[:, 1, 0], post_rot[:, 1, 1], post_tran[:, 1],
         bda[:, 0, 0], bda[:, 0, 1], bda[:, 1, 0], bda[:, 1, 1], bda[:, 2, 2]]
    return torch.cat([torch.stack(v, 1), sensor2ego[:, :3, :4].reshape(B, 12)], 1)


def nuscenes_front_ida(src_hw=(900, 1600), dst_hw=(256, 704)):
    """DualBEV test-time image transform: resize by dst_w / src_w, crop the top rows -> 3x3 matrix (pixels)."""
    s = dst_hw[1] / src_hw[1]
    crop_top = int(round(src_hw[0] * s)) - dst_hw[0]
    return torch.tensor([[s, 0, 0], [0, s, -crop_top], [0, 0, 1]], dtype=torch.float32)


def bevfusion_ida(scale=0.48, crop=(32, 176)):
    """3 x 3 pixel transform original -> augmented image (test pipeline: scale, crop x0 / y0)."""
    return torch.tensor([[scale, 0, -crop[0]], [0, scale, -crop[1]], [0, 0, 1]], dtype=torch.float32)


BEVFUSION_MEAN, BEVFUSION_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)             # the BEVFusion pipeline normalises on 0..1


def load_image(path, scale=0.48, crop=(32, 176, 736, 432)):
    """BEVFusion test-time image: PIL bicubic resize, crop, ImageNet normalisation -> (3, 256, 704)."""
    import numpy as np
    from PIL import Image
    img = Image.open(path).convert("RGB")
    img = img.resize((int(img.width * scale), int(img.height * scale)), Image.BICUBIC).crop(crop)
    x = torch.from_numpy(np.asarray(img).copy()).float().div(255).permute(2, 0, 1)
    return (x - torch.tensor(BEVFUSION_MEAN).view(3, 1, 1)) / torch.tensor(BEVFUSION_STD).view(3, 1, 1)


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
