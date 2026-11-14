"""Train the project GPT directly from a Hugging Face streaming dataset."""

import ast
from contextlib import nullcontext
from itertools import islice
from pathlib import Path
import sys

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import Config, TrainConfig
from gpt import GPT
from optimizer import build_optimizer, learning_rate_at, set_learning_rate


# Dataset rows contain Python source in the "content" field.
TRAIN_DATASET = "codeparrot/codeparrot-clean"
VALID_DATASET = "codeparrot/codeparrot-clean-valid"
TOKENIZER_PATH = ROOT / "tokenizer" / "code_bpe_16k"

BLOCK_SIZE = 512
BATCH_SIZE = 16
MAX_STEPS = 10_000
EVAL_EVERY = 200
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
    """Convert streamed source files into next-token prediction batches."""
    batch_x = []
    batch_y = []
    eos_id = tokenizer.eos_token_id

    for row in dataset:
        code = row.get("content")
        if not isinstance(code, str) or not code.strip():
            continue

        # Skip files that are not valid Python source.
        try:
            ast.parse(code)
        except (SyntaxError, ValueError, RecursionError):
            continue

        token_ids = tokenizer.encode(code, add_special_tokens=False)
        if eos_id is not None:
            token_ids.append(eos_id)

        # Each window has block_size inputs and the following token as targets.
        for start in range(0, len(token_ids) - block_size, block_size):
            window = token_ids[start : start + block_size + 1]
            if len(window) != block_size + 1:
                continue

            batch_x.append(window[:-1])
            batch_y.append(window[1:])

            if len(batch_x) == batch_size:
                yield (
                    torch.tensor(batch_x, dtype=torch.long),
                    torch.tensor(batch_y, dtype=torch.long),
                )
                batch_x = []
                batch_y = []

    if batch_x:
        yield (
            torch.tensor(batch_x, dtype=torch.long),
            torch.tensor(batch_y, dtype=torch.long),
        )


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


def main():
    torch.manual_seed(SEED)
    device = get_device()
    print(f"device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH))
    print(f"vocab size: {len(tokenizer):,}")

    train_data = load_dataset(
        TRAIN_DATASET,
        split="train",
        streaming=True,
    ).shuffle(seed=SEED, buffer_size=1_000)

    # This repository contains a separate validation collection (its split is named "train").
    valid_data = load_dataset(
        VALID_DATASET,
        split="train",
        streaming=True,
    ).shuffle(seed=SEED + 1, buffer_size=1_000)

    valid_batches = list(
        islice(
            stream_batches(valid_data, tokenizer, BLOCK_SIZE, BATCH_SIZE),
            EVAL_BATCHES,
        )
    )
    if not valid_batches:
        raise RuntimeError("검증 배치를 만들지 못했습니다. 데이터셋의 content 필드를 확인하세요.")

    model_config = Config(
        vocab_size=len(tokenizer),
        block_size=BLOCK_SIZE,
        n_layer=16,
        n_embd=768,
        n_head=12,
        dropout=0.1,
    )
    model = GPT(model_config).to(device)

    train_config = TrainConfig(
        max_iters=MAX_STEPS,
        batch_size=BATCH_SIZE,
        eval_every=EVAL_EVERY,
    )
    optimizer = build_optimizer(model, train_config)
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=device.type in {"cuda", "mps"},
    )

    epoch = 0
    train_data.set_epoch(epoch)
    train_batches = iter(
        stream_batches(train_data, tokenizer, BLOCK_SIZE, BATCH_SIZE)
    )

    model.train()
    for step in range(MAX_STEPS):
        try:
            x, y = next(train_batches)
        except StopIteration:
            # Start another streamed pass with a new shuffle order.
            epoch += 1
            train_data.set_epoch(epoch)
            train_batches = iter(
                stream_batches(train_data, tokenizer, BLOCK_SIZE, BATCH_SIZE)
            )
            x, y = next(train_batches)

        x, y = x.to(device), y.to(device)
        set_learning_rate(optimizer, step, train_config)
        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device):
            _, loss = model(x, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        completed_steps = step + 1
        if completed_steps % LOG_EVERY == 0:
            lr = learning_rate_at(step, train_config)
            print(
                f"step {completed_steps:>6} | "
                f"train loss {loss.item():.4f} | lr {lr:.2e}"
            )

        if completed_steps % EVAL_EVERY == 0:
            val_loss = evaluate(model, valid_batches, device)
            print(f"step {completed_steps:>6} | val loss {val_loss:.4f}")


if __name__ == "__main__":
    main()
