import argparse
import math
import os
import random
import sysconfig
import time
from pathlib import Path

import torch

from config import Config
from gpt import GPT


ROOT = Path(__file__).resolve().parent
SKIP_DIRS = {"__pycache__", "site-packages", "dist-packages"}


def python_files(root):
    for directory, directories, filenames in os.walk(root):
        directories[:] = sorted(name for name in directories if name not in SKIP_DIRS)
        for filename in sorted(filenames):
            if filename.endswith(".py"):
                yield Path(directory) / filename


def load_corpus(root):
    root = Path(root).expanduser()
    if not root.is_absolute():
        root = ROOT / root
    if not root.is_dir():
        raise SystemExit(f"파이썬 소스 폴더를 찾을 수 없습니다: {root}")

    files = []
    for path in python_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if text:
            files.append(text)

    if len(files) < 2:
        raise SystemExit(f"학습에 사용할 UTF-8 파이썬 파일이 부족합니다: {root}")

    random.Random(0).shuffle(files)
    n_val = max(1, int(len(files) * 0.1))
    n_val = min(n_val, len(files) - 1)
    val_text = "\n\n".join(files[:n_val])
    train_text = "\n\n".join(files[n_val:])

    chars = sorted(set(train_text + val_text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    train_data = torch.tensor([stoi[ch] for ch in train_text], dtype=torch.long)
    val_data = torch.tensor([stoi[ch] for ch in val_text], dtype=torch.long)
    return chars, train_data, val_data, len(files)


def get_batch(data, batch_size, block_size, device):
    starts = torch.randint(data.numel() - block_size, (batch_size,))
    offsets = torch.arange(block_size + 1)
    windows = data[starts[:, None] + offsets[None, :]]
    return windows[:, :-1].to(device), windows[:, 1:].to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size, eval_iters, device):
    model.eval()
    losses = {}
    for split, data in (("train", train_data), ("val", val_data)):
        values = []
        for _ in range(eval_iters):
            x, y = get_batch(data, batch_size, block_size, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                _, loss = model(x, y)
            values.append(loss.item())
        losses[split] = sum(values) / len(values)
    model.train()
    return losses


def get_lr(iteration, max_iters, lr, min_lr, warmup):
    if iteration < warmup:
        return lr * (iteration + 1) / warmup
    ratio = (iteration - warmup) / max(1, max_iters - warmup)
    ratio = min(1.0, max(0.0, ratio))
    return min_lr + 0.5 * (1 + math.cos(math.pi * ratio)) * (lr - min_lr)


@torch.no_grad()
def generate(model, prompt, chars, stoi, block_size, device, max_new_tokens=200):
    ids = [stoi[ch] for ch in prompt]
    index = torch.tensor([ids], dtype=torch.long, device=device)
    model.eval()

    for _ in range(max_new_tokens):
        context = index[:, -block_size:]
        logits, _ = model(context)
        logits = logits[:, -1, :] / 0.8
        top_k = min(40, logits.size(-1))
        threshold = torch.topk(logits, top_k).values[:, [-1]]
        logits[logits < threshold] = float("-inf")
        probabilities = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)
        index = torch.cat((index, next_token), dim=1)

    return "".join(chars[i] for i in index[0].tolist())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the character-level GPT on Python source files."
    )
    parser.add_argument(
        "--corpus-dir",
        default=sysconfig.get_paths()["stdlib"],
        help="Python source directory (defaults to this environment's standard library).",
    )
    parser.add_argument("--max-iters", type=int, default=4000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--checkpoint", default="code_gpt_ckpt.pt")
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_iters < 0 or args.eval_every < 1 or args.eval_iters < 1:
        raise SystemExit("max-iters는 0 이상, eval-every와 eval-iters는 1 이상이어야 합니다.")
    if args.batch_size < 1 or args.block_size < 1:
        raise SystemExit("batch-size와 block-size는 1 이상이어야 합니다.")
    if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
        raise SystemExit("MPS를 사용할 수 없습니다. Apple Silicon의 transformer 환경에서 실행하세요.")

    device = torch.device("mps")
    torch.manual_seed(args.seed)
    corpus_dir = Path(args.corpus_dir).expanduser()
    if not corpus_dir.is_absolute():
        corpus_dir = ROOT / corpus_dir
    chars, train_data, val_data, file_count = load_corpus(corpus_dir)
    block_size = min(args.block_size, train_data.numel() - 1, val_data.numel() - 1)
    if block_size < 1:
        raise SystemExit("학습/검증 데이터가 너무 짧습니다.")

    print(f"device: {device}")
    print(f"corpus: {corpus_dir}")
    print(f"files: {file_count}, chars: {train_data.numel() + val_data.numel():,}, vocab: {len(chars)}")
    if block_size != args.block_size:
        print(f"block_size를 데이터 길이에 맞춰 {block_size}로 줄였습니다.")

    cfg = Config(vocab_size=len(chars), block_size=block_size)
    model = GPT(cfg).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 0.1},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=6e-4,
        betas=(0.9, 0.95),
    )
    scaler = torch.amp.GradScaler(device.type)

    checkpoint = Path(args.checkpoint).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    last_checkpoint = checkpoint.with_name(f"{checkpoint.stem}_last{checkpoint.suffix}")

    best_val = float("inf")
    started = time.perf_counter()
    for iteration in range(args.max_iters + 1):
        lr = get_lr(iteration, args.max_iters, 6e-4, 6e-5, 100)
        for group in opt.param_groups:
            group["lr"] = lr

        should_evaluate = iteration % args.eval_every == 0 or iteration == args.max_iters
        if should_evaluate:
            losses = estimate_loss(
                model,
                train_data,
                val_data,
                args.batch_size,
                block_size,
                args.eval_iters,
                device,
            )
            elapsed = time.perf_counter() - started
            print(
                f"it {iteration:5d} | train {losses['train']:.3f} "
                f"| val {losses['val']:.3f} | {elapsed:.0f}s"
            )
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "cfg": cfg.__dict__,
                        "it": iteration,
                        "chars": chars,
                        "val": best_val,
                    },
                    checkpoint,
                )

        if iteration == args.max_iters:
            break

        x, y = get_batch(train_data, args.batch_size, block_size, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

    torch.save(
        {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "cfg": cfg.__dict__,
            "chars": chars,
        },
        last_checkpoint,
    )
    print(f"saved best: {checkpoint}")
    print(f"saved last: {last_checkpoint}")
    print("sample:")
    print(generate(model, "def ", chars, {ch: i for i, ch in enumerate(chars)}, block_size, device))


if __name__ == "__main__":
    main()
