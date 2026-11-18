"""Mixed precision: the autocast context of the forward pass and the GradScaler of amp "fp16"."""
import torch

AMP_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def amp_context(amp, device):
    """the autocast context of the forward pass: bf16 / fp16 on CUDA, nothing otherwise. An unknown name is an error (it used to mean `off`)"""
    if amp != "off" and amp not in AMP_DTYPES:
        raise ValueError(f"amp must be one of off, {', '.join(AMP_DTYPES)}; got {amp!r}")
    if amp in AMP_DTYPES and device.startswith("cuda"):
        return torch.autocast("cuda", dtype=AMP_DTYPES[amp])
    return torch.autocast("cpu", enabled=False)


def make_scaler(cfg, device):
    """amp 'fp16': a GradScaler (the fp16 range is small: gradients below ~6e-8 vanish, values above 65504 overflow, so the loss is scaled and the gradients are
    unscaled before the clip); None otherwise (bf16 has the range of fp32). An unknown amp name is an error here, before the model and the data are built.
    NOT covered: an overflow in the FORWARD pass (activations above 65504) is no scaler event, it gives a non-finite loss and leaves non-finite batch-norm statistics (see optim.poisoned_buffers)"""
    amp_context(cfg.amp, device)
    if cfg.amp != "fp16":
        return None
    return torch.amp.GradScaler("cuda" if device.startswith("cuda") else "cpu", init_scale=cfg.fp16_scale, growth_interval=cfg.fp16_growth_interval)


def scaler_extra(scaler):
    """the part of a checkpoint's `extra` that holds the scaler (its scale and growth counter), {} without one"""
    return {"scaler": scaler.state_dict()} if scaler is not None else {}
