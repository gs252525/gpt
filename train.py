import argparse
from dataclasses import asdict
from pathlib import Path
import time

import torch

from checkpoint import (
    FORMAT_VERSION,
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from config import Config, DataConfig, TrainConfig
from data.data import encode_splits, load_text_splits, make_batch_source
from gpt import GPT
from optimizer import build_optimizer, set_learning_rate
from tokenizer import make_tokenizer


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Train GPT with a selectable tokenizer and batch source.")
    parser.add_argument("--corpus-dir", help="Python source directory or a UTF-8 text file.")
    parser.add_argument("--batcher", choices=("random", "dataloader"))
    parser.add_argument("--tokenizer", choices=("char", "hf-bpe"))
    parser.add_argument("--tokenizer-name", help="Hugging Face tokenizer id or local directory (default: gpt2).")
    parser.add_argument("--tokenizer-revision", help="Hugging Face branch, tag, or commit id.")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--n-layer", type=int)
    parser.add_argument("--n-head", type=int)
    parser.add_argument("--n-embd", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--max-iters", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--eval-iters", type=int)
    parser.add_argument("--save-every", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--min-lr", type=float)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--grad-clip", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--beta1", type=float)
    parser.add_argument("--beta2", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--checkpoint", help="Path for the best-model checkpoint.")
    parser.add_argument("--resume", help="Resume from a full *_last.pt training checkpoint.")
    parser.add_argument("--init-from", help="Start with model weights from a best-model checkpoint.")
    return parser.parse_args()


def _project_path(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return str(path.resolve())


def _new_configs(args):
    model_config = Config(vocab_size=0)
    data_config = DataConfig()
    train_config = TrainConfig()

    for arg_name, config_name in (
        ("block_size", "block_size"),
        ("n_layer", "n_layer"),
        ("n_head", "n_head"),
        ("n_embd", "n_embd"),
        ("dropout", "dropout"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            setattr(model_config, config_name, value)

    if args.corpus_dir is not None:
        data_config.corpus_dir = _project_path(args.corpus_dir)
    if args.batcher is not None:
        data_config.batcher = args.batcher
    if args.tokenizer is not None:
        data_config.tokenizer = args.tokenizer
    if args.tokenizer_name is not None:
        data_config.tokenizer_name = args.tokenizer_name
    if args.tokenizer_revision is not None:
        data_config.tokenizer_revision = args.tokenizer_revision
    if args.split_seed is not None:
        data_config.split_seed = args.split_seed

    for name in (
        "max_iters",
        "batch_size",
        "eval_every",
        "eval_iters",
        "save_every",
        "lr",
        "min_lr",
        "warmup",
        "grad_clip",
        "weight_decay",
        "beta1",
        "beta2",
        "seed",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(train_config, name, value)

    return model_config, data_config, train_config


def _configs_from_resume(args, state):
    if state.get("format_version") not in (1, FORMAT_VERSION) or state.get("kind") != "resume":
        raise SystemExit("--resume에는 새 형식의 전체 학습 체크포인트(*_last.pt)를 지정하세요.")

    model_config = Config(**state["model_config"])
    data_config = DataConfig(**state["data_config"])
    train_config = TrainConfig(**state["train_config"])

    for name in ("batcher", "tokenizer", "tokenizer_name", "tokenizer_revision"):
        value = getattr(args, name)
        if value is not None and value != getattr(data_config, name):
            raise SystemExit(f"resume할 때 {name}은 체크포인트 값 {getattr(data_config, name)!r}을 사용하세요.")
    if args.batch_size is not None and args.batch_size != train_config.batch_size:
        raise SystemExit("resume할 때 batch-size는 체크포인트 값과 같아야 합니다.")
    if args.block_size is not None and args.block_size != model_config.block_size:
        raise SystemExit("resume할 때 block-size는 체크포인트 값과 같아야 합니다.")
    for name in ("n_layer", "n_head", "n_embd", "dropout"):
        value = getattr(args, name)
        if value is not None and value != getattr(model_config, name):
            raise SystemExit(f"resume할 때 {name}은 체크포인트 값과 같아야 합니다.")
    if args.seed is not None and args.seed != train_config.seed:
        raise SystemExit("resume할 때 seed는 체크포인트 값과 같아야 합니다.")
    if args.split_seed is not None and args.split_seed != data_config.split_seed:
        raise SystemExit("resume할 때 split-seed는 체크포인트 값과 같아야 합니다.")

    if args.corpus_dir is not None:
        data_config.corpus_dir = _project_path(args.corpus_dir)

    for name in (
        "max_iters",
        "eval_every",
        "eval_iters",
        "save_every",
        "lr",
        "min_lr",
        "warmup",
        "grad_clip",
        "weight_decay",
        "beta1",
        "beta2",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(train_config, name, value)

    return model_config, data_config, train_config


def _validate_config(model_config, train_config, start_step=0):
    if model_config.block_size < 1 or model_config.n_layer < 1 or model_config.n_head < 1:
        raise SystemExit("block-size, n-layer, n-head는 1 이상이어야 합니다.")
    if model_config.n_embd < 1 or model_config.n_embd % model_config.n_head != 0:
        raise SystemExit("n-embd는 양수이고 n-head로 나누어져야 합니다.")
    if train_config.max_iters < start_step:
        raise SystemExit("max-iters는 resume step보다 작을 수 없습니다.")
    if train_config.max_iters < 0 or train_config.batch_size < 1:
        raise SystemExit("max-iters는 0 이상, batch-size는 1 이상이어야 합니다.")
    if train_config.eval_every < 1 or train_config.eval_iters < 1 or train_config.save_every < 1:
        raise SystemExit("eval-every, eval-iters, save-every는 1 이상이어야 합니다.")
    if train_config.warmup < 0 or train_config.lr <= 0 or train_config.min_lr < 0:
        raise SystemExit("learning rate와 warmup 설정을 확인하세요.")


def _checkpoint_paths(args, resume_state):
    if args.checkpoint is not None:
        best_path = Path(_project_path(args.checkpoint))
    elif resume_state is not None:
        resume_path = Path(_project_path(args.resume))
        best_name = resume_state.get("best_checkpoint_name")
        if best_name:
            best_path = resume_path.parent / best_name
        else:
            stem = resume_path.stem.removesuffix("_last")
            best_path = resume_path.with_name(stem + resume_path.suffix)
    else:
        best_path = ROOT / "code_gpt_ckpt.pt"

    last_path = best_path.with_name(f"{best_path.stem}_last{best_path.suffix}")
    return best_path, last_path


def estimate_loss(model, batch_source, device, eval_iters):
    model.eval()
    losses = {}
    try:
        with torch.no_grad():
            for split in ("train", "val"):
                values = []
                for _ in range(eval_iters):
                    x, y = batch_source.next_batch(split)
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device.type, dtype=torch.float16):
                        _, loss = model(x, y)
                    values.append(loss.item())
                losses[split] = sum(values) / len(values)
    finally:
        model.train()
    return losses


def generate(model, tokenizer, block_size, device, prompt, max_new_tokens=200):
    index = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = index[:, -block_size:]
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits, _ = model(context)
            logits = logits[:, -1, :] / 0.8
            top_k = min(40, logits.size(-1))
            threshold = torch.topk(logits, top_k).values[:, [-1]]
            logits[logits < threshold] = float("-inf")
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            index = torch.cat((index, next_token), dim=1)
    return tokenizer.decode(index[0].tolist())


def _resume_state(
    next_step,
    best_val,
    model,
    optimizer,
    scaler,
    batch_source,
    model_config,
    train_config,
    data_config,
    tokenizer,
    data_fingerprint,
    best_path,
):
    return {
        "format_version": FORMAT_VERSION,
        "kind": "resume",
        "step": next_step,
        "best_val": best_val,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "data_config": asdict(data_config),
        "tokenizer": tokenizer.state_dict(),
        "batch_source": batch_source.state_dict(),
        "data_fingerprint": data_fingerprint,
        "rng": capture_rng_state(),
        "best_checkpoint_name": best_path.name,
    }


def _save_best(path, step, val_loss, model, model_config, tokenizer):
    save_checkpoint(
        path,
        {
            "format_version": FORMAT_VERSION,
            "kind": "weights",
            "step": step,
            "best_val": val_loss,
            "model": model.state_dict(),
            "model_config": asdict(model_config),
            "tokenizer": tokenizer.state_dict(),
        },
    )


def main():
    args = parse_args()
    if args.resume and args.init_from:
        raise SystemExit("--resume와 --init-from은 함께 사용할 수 없습니다.")

    resume_state = load_checkpoint(_project_path(args.resume)) if args.resume else None
    init_state = load_checkpoint(_project_path(args.init_from)) if args.init_from else None

    if resume_state is not None:
        model_config, data_config, train_config = _configs_from_resume(args, resume_state)
        tokenizer = make_tokenizer(
            data_config.tokenizer,
            state=resume_state["tokenizer"],
            pretrained_name=data_config.tokenizer_name,
            revision=data_config.tokenizer_revision,
        )
        data_config.tokenizer_revision = getattr(tokenizer, "revision", data_config.tokenizer_revision)
        start_step = int(resume_state["step"])
        best_val = float(resume_state["best_val"])
    else:
        model_config, data_config, train_config = _new_configs(args)
        if init_state is not None:
            saved_model_config = init_state.get("model_config", init_state.get("cfg"))
            if saved_model_config is None:
                raise SystemExit("--init-from 체크포인트에 model_config가 없습니다.")
            model_config = Config(**saved_model_config)
            for name in ("block_size", "n_layer", "n_head", "n_embd", "dropout"):
                value = getattr(args, name)
                if value is not None and value != getattr(model_config, name):
                    raise SystemExit(f"init-from 체크포인트와 {name}이 다릅니다.")
            tokenizer_state = init_state.get("tokenizer")
            if tokenizer_state is None and "chars" in init_state:
                tokenizer_state = {"type": "char", "chars": init_state["chars"]}
            if tokenizer_state is None:
                raise SystemExit("--init-from 체크포인트에 tokenizer 상태가 없습니다.")
            data_config.tokenizer = tokenizer_state["type"]
            data_config.tokenizer_name = tokenizer_state.get("model_name", data_config.tokenizer_name)
            data_config.tokenizer_revision = tokenizer_state.get("revision")
            tokenizer = make_tokenizer(
                data_config.tokenizer,
                state=tokenizer_state,
                pretrained_name=data_config.tokenizer_name,
                revision=data_config.tokenizer_revision,
            )
        else:
            tokenizer = None
        start_step = 0
        best_val = float("inf")

    _validate_config(model_config, train_config, start_step)
    if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
        raise SystemExit("MPS를 사용할 수 없습니다. Apple Silicon의 transformer 환경에서 실행하세요.")
    device = torch.device("mps")
    torch.manual_seed(train_config.seed)

    splits = load_text_splits(
        data_config.corpus_dir,
        train_fraction=data_config.train_fraction,
        seed=data_config.split_seed,
    )
    if resume_state is not None and splits.fingerprint != resume_state["data_fingerprint"]:
        raise SystemExit("코퍼스가 체크포인트를 만든 뒤 변경되었습니다. 같은 데이터를 지정하세요.")
    if tokenizer is None:
        tokenizer = make_tokenizer(
            data_config.tokenizer,
            text=splits.train_text + splits.val_text,
            pretrained_name=data_config.tokenizer_name,
            revision=data_config.tokenizer_revision,
        )
        data_config.tokenizer_revision = getattr(tokenizer, "revision", data_config.tokenizer_revision)
    if tokenizer.vocab_size != model_config.vocab_size and model_config.vocab_size != 0:
        raise SystemExit("tokenizer vocabulary와 모델 체크포인트가 맞지 않습니다.")
    model_config.vocab_size = tokenizer.vocab_size

    train_data, val_data = encode_splits(splits, tokenizer)
    max_block_size = min(train_data.numel() - 1, val_data.numel() - 1)
    if max_block_size < 1:
        raise SystemExit("학습/검증 데이터가 너무 짧아 batch를 만들 수 없습니다.")
    if model_config.block_size > max_block_size:
        if resume_state is not None or init_state is not None:
            raise SystemExit("체크포인트의 block-size가 현재 데이터보다 큽니다.")
        model_config.block_size = max_block_size
        print(f"block_size를 데이터 길이에 맞춰 {model_config.block_size}로 줄였습니다.")

    print(f"device: {device}")
    print(f"corpus: {data_config.corpus_dir}")
    print(
        f"files: {splits.file_count}, chars: {len(splits.train_text) + len(splits.val_text):,}, "
        f"tokens: {train_data.numel() + val_data.numel():,}, vocab: {tokenizer.vocab_size}"
    )
    print(f"batcher: {data_config.batcher} | tokenizer: {data_config.tokenizer}")

    model = GPT(model_config).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    optimizer = build_optimizer(model, train_config)
    scaler = torch.amp.GradScaler(device.type)
    batch_source = make_batch_source(
        data_config.batcher,
        train_data,
        val_data,
        train_config.batch_size,
        model_config.block_size,
        train_config.seed,
    )

    if resume_state is not None:
        model.load_state_dict(resume_state["model"])
        optimizer.load_state_dict(resume_state["optimizer"])
        scaler.load_state_dict(resume_state["scaler"])
        batch_source.load_state_dict(resume_state["batch_source"])
        restore_rng_state(resume_state["rng"])
        print(f"resuming from step {start_step}")
    elif init_state is not None:
        model.load_state_dict(init_state["model"])
        print("initialized model weights; optimizer and step counter start fresh")

    best_path, last_path = _checkpoint_paths(args, resume_state)
    next_step = start_step
    started = time.perf_counter()

    def save_last():
        save_checkpoint(
            last_path,
            _resume_state(
                next_step,
                best_val,
                model,
                optimizer,
                scaler,
                batch_source,
                model_config,
                train_config,
                data_config,
                tokenizer,
                splits.fingerprint,
                best_path,
            ),
        )

    try:
        while next_step < train_config.max_iters:
            step = next_step
            set_learning_rate(optimizer, step, train_config)
            if step % train_config.eval_every == 0:
                losses = estimate_loss(
                    model,
                    batch_source,
                    device,
                    train_config.eval_iters,
                )
                print(
                    f"it {step:5d} | train {losses['train']:.3f} "
                    f"| val {losses['val']:.3f} | {time.perf_counter() - started:.0f}s"
                )
                if losses["val"] < best_val:
                    best_val = losses["val"]
                    _save_best(best_path, step, best_val, model, model_config, tokenizer)

            x, y = batch_source.next_batch("train")
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            next_step = step + 1

            if next_step % train_config.save_every == 0:
                save_last()
                print(f"saved resume checkpoint at step {next_step}: {last_path}")

        batch_source_state = batch_source.state_dict()
        rng_state = capture_rng_state()
        losses = estimate_loss(
            model,
            batch_source,
            device,
            train_config.eval_iters,
        )
        print(
            f"it {next_step:5d} | train {losses['train']:.3f} "
            f"| val {losses['val']:.3f} | {time.perf_counter() - started:.0f}s"
        )
        if losses["val"] < best_val:
            best_val = losses["val"]
            _save_best(best_path, next_step, best_val, model, model_config, tokenizer)
        batch_source.load_state_dict(batch_source_state)
        restore_rng_state(rng_state)
        save_last()
    except KeyboardInterrupt:
        save_last()
        print(f"interrupted at step {next_step}; resume with --resume {last_path}")
        return

    print(f"saved best: {best_path}")
    print(f"saved resume state: {last_path}")
    prompt = splits.train_text[: min(4, len(splits.train_text))]
    if prompt:
        print("sample:")
        print(generate(model, tokenizer, model_config.block_size, device, prompt))


if __name__ == "__main__":
    main()
