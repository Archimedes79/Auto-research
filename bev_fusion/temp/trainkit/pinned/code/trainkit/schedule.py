"""The learning-rate schedule: linear warm-up from lr / 3, then cosine to lr_min_frac; a group of `Config.lr_groups` has its own peak lr and warm-up."""
import math


def lr_factor(step, warmup, total, min_frac):
    if step < warmup:
        return 1 / 3 + (2 / 3) * (step + 1) / warmup                      # from lr / 3 to lr
    t = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * t))


def scheduled_lr(group, step, cfg, f):
    """the learning rate of one optimiser group at `step`; f = lr_factor with the chain's warm-up. A group without the tags of lr_groups gets cfg.lr * f (the single schedule, bit for bit);
    a tagged one its own peak lr (base_lr) and, if it has one, its own warm-up"""
    if "base_lr" not in group:
        return cfg.lr * f
    return group["base_lr"] * (f if "warmup" not in group else lr_factor(step, group["warmup"], cfg.steps, cfg.lr_min_frac))
