import hashlib
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parent
SKIP_DIRS = {"__pycache__", "site-packages", "dist-packages"}


@dataclass
class TextSplits:
    train_text: str
    val_text: str
    fingerprint: str
    file_count: int


class BatchSource(Protocol):
    def next_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]: ...

    def state_dict(self) -> dict: ...

    def load_state_dict(self, state: dict) -> None: ...


def _python_files(root: Path):
    for directory, directories, filenames in os.walk(root):
        directories[:] = sorted(name for name in directories if name not in SKIP_DIRS)
        for filename in sorted(filenames):
            if filename.endswith(".py"):
                yield Path(directory) / filename


def load_text_splits(corpus_path: str | Path, train_fraction: float, seed: int) -> TextSplits:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")

    root = Path(corpus_path).expanduser()
    if not root.is_absolute():
        root = ROOT / root

    if root.is_file():
        try:
            records = [(root.name, root.read_text(encoding="utf-8"))]
        except (OSError, UnicodeDecodeError) as exc:
            raise SystemExit(f"텍스트 파일을 읽을 수 없습니다: {root}: {exc}") from exc
        if len(records[0][1]) < 2:
            raise SystemExit(f"학습/검증으로 나누기에는 파일이 너무 짧습니다: {root}")
        split_at = int(len(records[0][1]) * train_fraction)
        split_at = min(max(split_at, 1), len(records[0][1]) - 1)
        train_text = records[0][1][:split_at]
        val_text = records[0][1][split_at:]
    elif root.is_dir():
        records = []
        for path in _python_files(root):
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if text:
                records.append((path.relative_to(root).as_posix(), text))

        if len(records) < 2:
            raise SystemExit(f"학습에 사용할 UTF-8 파이썬 파일이 부족합니다: {root}")

        random.Random(seed).shuffle(records)
        n_val = max(1, int(len(records) * (1.0 - train_fraction)))
        n_val = min(n_val, len(records) - 1)
        val_text = "\n\n".join(text for _, text in records[:n_val])
        train_text = "\n\n".join(text for _, text in records[n_val:])
    else:
        raise SystemExit(f"코퍼스를 찾을 수 없습니다: {root}")

    digest = hashlib.sha256()
    for name, text in sorted(records, key=lambda item: item[0]):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")

    return TextSplits(
        train_text=train_text,
        val_text=val_text,
        fingerprint=digest.hexdigest(),
        file_count=len(records),
    )


def encode_splits(splits: TextSplits, tokenizer) -> tuple[torch.Tensor, torch.Tensor]:
    train_data = torch.tensor(tokenizer.encode(splits.train_text), dtype=torch.long)
    val_data = torch.tensor(tokenizer.encode(splits.val_text), dtype=torch.long)
    return train_data, val_data


class RandomBatchSource:
    """Vectorized random windows matching the original get_batch behavior."""

    name = "random"

    def __init__(self, train_data, val_data, batch_size, block_size, seed):
        self.data = {"train": train_data, "val": val_data}
        self.batch_size = batch_size
        self.block_size = block_size
        self.generators = {
            "train": torch.Generator().manual_seed(seed),
            "val": torch.Generator().manual_seed(seed + 1),
        }
        self._validate_data()

    def _validate_data(self):
        for split, data in self.data.items():
            if data.numel() <= self.block_size:
                raise ValueError(f"{split} data must contain more tokens than block_size")

    def next_batch(self, split):
        data = self.data[split]
        starts = torch.randint(
            data.numel() - self.block_size,
            (self.batch_size,),
            generator=self.generators[split],
        )
        offsets = torch.arange(self.block_size + 1)
        windows = data[starts[:, None] + offsets[None, :]]
        return windows[:, :-1], windows[:, 1:]

    def state_dict(self):
        return {
            "kind": self.name,
            "batch_size": self.batch_size,
            "block_size": self.block_size,
            "generators": {split: gen.get_state() for split, gen in self.generators.items()},
        }

    def load_state_dict(self, state):
        _validate_batch_source_state(state, self.name, self.batch_size, self.block_size)
        for split, generator_state in state["generators"].items():
            self.generators[split].set_state(generator_state)


class WindowDataset(Dataset):
    def __init__(self, tokens, block_size):
        self.tokens = tokens
        self.block_size = block_size
        if tokens.numel() <= block_size:
            raise ValueError("dataset must contain more tokens than block_size")

    def __len__(self):
        return self.tokens.numel() - self.block_size

    def __getitem__(self, start):
        end = start + self.block_size
        return self.tokens[start:end], self.tokens[start + 1 : end + 1]


class DataLoaderBatchSource:
    """Shuffled window epochs; num_workers=0 keeps checkpoint resume reproducible."""

    name = "dataloader"

    def __init__(self, train_data, val_data, batch_size, block_size, seed):
        self.datasets = {
            "train": WindowDataset(train_data, block_size),
            "val": WindowDataset(val_data, block_size),
        }
        self.batch_size = batch_size
        self.block_size = block_size
        self.seed = seed
        self.epochs = {"train": 0, "val": 0}
        self.offsets = {"train": 0, "val": 0}
        self.iterators = {}
        self.batches_per_epoch = {
            split: len(dataset) // batch_size for split, dataset in self.datasets.items()
        }
        if any(count == 0 for count in self.batches_per_epoch.values()):
            raise ValueError("each split must contain at least one full DataLoader batch")

    def _build_iterator(self, split):
        split_number = 0 if split == "train" else 1
        generator = torch.Generator().manual_seed(
            self.seed + self.epochs[split] * 2 + split_number
        )
        loader = DataLoader(
            self.datasets[split],
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
            generator=generator,
        )
        iterator = iter(loader)
        for _ in range(self.offsets[split]):
            next(iterator)
        self.iterators[split] = iterator

    def next_batch(self, split):
        if split not in self.datasets:
            raise ValueError(f"unknown split: {split}")
        if split not in self.iterators:
            self._build_iterator(split)

        try:
            batch = next(self.iterators[split])
        except StopIteration:
            self.epochs[split] += 1
            self.offsets[split] = 0
            self.iterators.pop(split, None)
            return self.next_batch(split)

        self.offsets[split] += 1
        if self.offsets[split] == self.batches_per_epoch[split]:
            self.epochs[split] += 1
            self.offsets[split] = 0
            self.iterators.pop(split, None)
        return batch

    def state_dict(self):
        return {
            "kind": self.name,
            "batch_size": self.batch_size,
            "block_size": self.block_size,
            "seed": self.seed,
            "epochs": self.epochs.copy(),
            "offsets": self.offsets.copy(),
        }

    def load_state_dict(self, state):
        _validate_batch_source_state(state, self.name, self.batch_size, self.block_size)
        if state["seed"] != self.seed:
            raise ValueError("DataLoader seed differs from the checkpoint")
        self.epochs = dict(state["epochs"])
        self.offsets = dict(state["offsets"])
        self.iterators.clear()


def _validate_batch_source_state(state, name, batch_size, block_size):
    if state.get("kind") != name:
        raise ValueError(f"checkpoint batcher is {state.get('kind')!r}, requested {name!r}")
    if state.get("batch_size") != batch_size or state.get("block_size") != block_size:
        raise ValueError("batch_size or block_size differs from the checkpoint")


def make_batch_source(kind, train_data, val_data, batch_size, block_size, seed):
    providers = {
        RandomBatchSource.name: RandomBatchSource,
        DataLoaderBatchSource.name: DataLoaderBatchSource,
    }
    try:
        provider = providers[kind]
    except KeyError as exc:
        raise ValueError(f"unsupported batcher: {kind}") from exc
    return provider(train_data, val_data, batch_size, block_size, seed)
