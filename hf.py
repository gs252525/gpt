"""Continue training a project GPT from streamed CodeParrot Python files.

First run: load the existing weights-only 2,600-step checkpoint.
Later runs: resume model, optimizer, GradScaler, step and streaming cursor.
"""

import ast
from contextlib import nullcontext
from dataclasses import asdict
from itertools import islice
from pathlib import Path
import sys

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

# Works whether this file lives in the project root or a nested scripts folder.
ROOT = next(
    (p for p in Path(__file__).resolve().parents
     if (p / "config.py").is_file() and (p / "gpt.py").is_file()),
    None,
)
if ROOT is None:
    raise RuntimeError("Could not find project root containing config.py and gpt.py")
sys.path.insert(0, str(ROOT))

from config import Config, TrainConfig
from gpt import GPT
from optimizer import build_optimizer, learning_rate_at, set_learning_rate

TRAIN_DATASET = "codeparrot/codeparrot-clean"
VALID_DATASET = "codeparrot/codeparrot-clean-valid"
TOKENIZER_PATH = ROOT / "tokenizer" / "code_bpe_16k"

# Place the weights-only .pt file from train.ipynb at the project root.
INITIAL_WEIGHTS = ROOT / "1113-2600iter-val-loss-3.73.pt"
CHECKPOINT = ROOT / "checkpoints" / "codeparrot_resume.pt"
INITIAL_STEP = 2_600
MAX_STEPS = 10_000  # Cumulative steps, including the previous 2,600.

BLOCK_SIZE = 512
BATCH_SIZE = 16
EVAL_EVERY = 200
SAVE_EVERY = 2000
LOG_EVERY = 50
EVAL_BATCHES = 20
SEED = 42


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_context(device):
    if device.type in {"cuda", "mps"}:
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    return nullcontext()


def stream_batches(dataset, tokenizer, block_size, batch_size):
    """Simple stream for validation; drops each file's shorter final tail."""
    xs, ys = [], []
    for row in dataset:
        code = row.get("content")
        if not isinstance(code, str) or not code.strip():
            continue
        try:
            ast.parse(code)
        except (SyntaxError, ValueError, RecursionError):
            continue

        ids = tokenizer.encode(code, add_special_tokens=False)
        if tokenizer.eos_token_id is not None:
            ids.append(tokenizer.eos_token_id)

        for start in range(0, len(ids) - block_size, block_size):
            window = ids[start : start + block_size + 1]
            xs.append(window[:-1])
            ys.append(window[1:])
            if len(xs) == batch_size:
                yield torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)
                xs, ys = [], []
    if xs:
        yield torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


class StreamingBatcher:
    """Keep track of the current file and its unconsumed token windows.

    Hugging Face's iterable dataset tracks its row position. This wrapper also
    stores the already-tokenized current file and next window offset, preventing
    loss of the remaining windows from that file on resume.
    """

    def __init__(self, dataset, tokenizer, block_size, batch_size):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.batch_size = batch_size
        self.epoch = 0
        self.dataset.set_epoch(0)
        self.rows = iter(self.dataset)
        self.ids = []
        self.offset = 0

    def _window(self):
        while True:
            if self.offset + self.block_size < len(self.ids):
                window = self.ids[self.offset : self.offset + self.block_size + 1]
                self.offset += self.block_size
                return window

            try:
                row = next(self.rows)
            except StopIteration:
                self.epoch += 1
                self.dataset.set_epoch(self.epoch)
                self.rows = iter(self.dataset)
                row = next(self.rows)

            code = row.get("content")
            if not isinstance(code, str) or not code.strip():
                continue
            try:
                ast.parse(code)
            except (SyntaxError, ValueError, RecursionError):
                continue
            self.ids = self.tokenizer.encode(code, add_special_tokens=False)
            if self.tokenizer.eos_token_id is not None:
                self.ids.append(self.tokenizer.eos_token_id)
            self.offset = 0

    def next_batch(self):
        xs, ys = [], []
        for _ in range(self.batch_size):
            window = self._window()
            xs.append(window[:-1])
            ys.append(window[1:])
        return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)

    def state_dict(self):
        return {
            "epoch": self.epoch,
            "dataset": self.dataset.state_dict(),
            "ids": self.ids,
            "offset": self.offset,
        }

    def load_state_dict(self, state):
        self.epoch = int(state["epoch"])
        self.dataset.set_epoch(self.epoch)
        self.dataset.load_state_dict(state["dataset"])
        self.rows = iter(self.dataset)
        self.ids = state["ids"]
        self.offset = int(state["offset"])


@torch.no_grad()
def evaluate(model, batches, device):
    model.eval()
    losses = []
    try:
        for x, y in batches:
            x, y = x.to(device), y.to(device)
            with autocast_context(device):
                _, loss = model(x, y)
            losses.append(loss.item())
    finally:
        model.train()
    return sum(losses) / len(losses)


def save_checkpoint(step, model, optimizer, scaler, batcher, model_config):
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    tmp = CHECKPOINT.with_suffix(".tmp")
    package = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "stream": batcher.state_dict(),
        "model_config": asdict(model_config),
        "initial_step": INITIAL_STEP,
        "torch_rng": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        package["cuda_rng"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available() and hasattr(torch.mps, "get_rng_state"):
        package["mps_rng"] = torch.mps.get_rng_state()
    torch.save(package, tmp)
    tmp.replace(CHECKPOINT)
    print(f"[saved] step {step:,}: {CHECKPOINT}")


def restore_rng(package):
    if "torch_rng" in package:
        torch.set_rng_state(package["torch_rng"])
    if torch.cuda.is_available() and "cuda_rng" in package:
        torch.cuda.set_rng_state_all(package["cuda_rng"])
    if torch.backends.mps.is_available() and hasattr(torch.mps, "set_rng_state") and "mps_rng" in package:
        torch.mps.set_rng_state(package["mps_rng"])


def main():
    torch.manual_seed(SEED)
    device = get_device()
    print(f"device: {device}; batch size: {BATCH_SIZE}")

    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH))
    model_config = Config(
        vocab_size=len(tokenizer),
        block_size=BLOCK_SIZE,
        n_layer=16,
        n_embd=768,
        n_head=12,
        dropout=0.1,
    )
    model = GPT(model_config).to(device)

    # Restart the LR schedule after loading the weights-only checkpoint,
    # because the original optimizer and schedule states weren't saved.
    train_config = TrainConfig(
        max_iters=MAX_STEPS - INITIAL_STEP,
        batch_size=BATCH_SIZE,
        eval_every=EVAL_EVERY,
    )
    optimizer = build_optimizer(model, train_config)
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=device.type in {"cuda", "mps"},
    )

    train_data = load_dataset(
        TRAIN_DATASET, split="train", streaming=True
    ).shuffle(seed=SEED, buffer_size=1_000)
    batcher = StreamingBatcher(train_data, tokenizer, BLOCK_SIZE, BATCH_SIZE)

    # Save/load only checkpoints that you created or trust.
    if CHECKPOINT.is_file():
        package = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        if package.get("initial_step") != INITIAL_STEP:
            raise ValueError("Checkpoint and INITIAL_STEP disagree")
        if package.get("model_config") != asdict(model_config):
            raise ValueError("Checkpoint architecture/tokenizer size differs")
        model.load_state_dict(package["model"])
        optimizer.load_state_dict(package["optimizer"])
        scaler.load_state_dict(package["scaler"])
        batcher.load_state_dict(package["stream"])
        step = int(package["step"])
        restore_rng(package)
        print(f"[resume] full checkpoint at step {step:,}")
    else:
        if not INITIAL_WEIGHTS.is_file():
            raise FileNotFoundError(
                f"Missing {INITIAL_WEIGHTS}. Put the 2,600-step weights file "
                "at the project root, or adjust INITIAL_WEIGHTS."
            )
        model.load_state_dict(
            torch.load(INITIAL_WEIGHTS, map_location="cpu", weights_only=True)
        )
        step = INITIAL_STEP
        print(f"[resume] weights-only checkpoint at step {step:,}")
        print("[notice] optimizer and data stream start fresh this first time")

    if step >= MAX_STEPS:
        print(f"Already at step {step:,}, target {MAX_STEPS:,}. Nothing to train.")
        return

    valid_data = load_dataset(
        VALID_DATASET, split="train", streaming=True
    ).shuffle(seed=SEED + 1, buffer_size=1_000)
    valid_batches = list(islice(
        stream_batches(valid_data, tokenizer, BLOCK_SIZE, BATCH_SIZE),
        EVAL_BATCHES,
    ))
    if not valid_batches:
        raise RuntimeError("No validation batches produced; check the dataset")

    model.train()
    try : 
        while step < MAX_STEPS:
            x, y = batcher.next_batch()
            x, y = x.to(device), y.to(device)
            lr_step = step - INITIAL_STEP
            set_learning_rate(optimizer, lr_step, train_config)
            optimizer.zero_grad(set_to_none=True)

            with autocast_context(device):
                _, loss = model(x, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            step += 1
            if step % LOG_EVERY == 0:
                lr = learning_rate_at(lr_step, train_config)
                print(f"step {step:>6} | train loss {loss.item():.4f} | lr {lr:.2e}")
            if step % EVAL_EVERY == 0:
                val_loss = evaluate(model, valid_batches, device)
                print(f"step {step:>6} | val loss {val_loss:.4f}")
            if step % SAVE_EVERY == 0 or step == MAX_STEPS:
                save_checkpoint(step, model, optimizer, scaler, batcher, model_config)
    except KeyboardInterrupt:
        print('체크포인트 저장중') 

    finally : 
        save_checkpoint(step, model, optimizer, scaler, batcher, model_config)




if __name__ == "__main__":
    main()
