import math

import torch


def build_optimizer(model, config):
    decay = [parameter for parameter in model.parameters() if parameter.dim() >= 2]
    no_decay = [parameter for parameter in model.parameters() if parameter.dim() < 2]
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.lr,
        betas=(config.beta1, config.beta2),
    )


def learning_rate_at(step, config):
    if step < config.warmup:
        return config.lr * (step + 1) / config.warmup
    ratio = (step - config.warmup) / max(1, config.max_iters - config.warmup)
    ratio = min(1.0, max(0.0, ratio))
    return config.min_lr + 0.5 * (1 + math.cos(math.pi * ratio)) * (config.lr - config.min_lr)


def set_learning_rate(optimizer, step, config):
    learning_rate = learning_rate_at(step, config)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return learning_rate
