import os
from pathlib import Path

import torch


FORMAT_VERSION = 2


def save_checkpoint(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    torch.save(state, temporary_path)
    os.replace(temporary_path, path)


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def capture_rng_state():
    state = {"cpu": torch.get_rng_state()}
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state().cpu()
    return state


def restore_rng_state(state):
    torch.set_rng_state(state["cpu"])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])
