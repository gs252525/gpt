
import hashlib
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def text_chunks(path, chunk_chars=100_000):
    """텍스트를 줄 경계에서 적당한 크기로 분할."""
    with open(path, "r", encoding="utf-8") as f:
        lines = []
        size = 0

        for line in f:
            lines.append(line)
            size += len(line)

            if size >= chunk_chars:
                yield "".join(lines)
                lines = []
                size = 0

        if lines:
            yield "".join(lines)


def load_tokens(path, tokenizer):
    path = Path(path)

    # 토크나이저 또는 원본 파일이 변경되면 새 캐시 사용
    stat = path.stat()
    key = (
        tokenizer.backend_tokenizer.to_str()
        + str(stat.st_size)
        + str(stat.st_mtime_ns)
    )
    fingerprint = hashlib.sha256(
        key.encode()
    ).hexdigest()[:12]

    dtype = np.uint16 if len(tokenizer) <= 65536 else np.uint32

    cache_path = path.with_name(
        f"{path.stem}.{fingerprint}.bin"
    )

    if not cache_path.exists():
        print(f"Tokenizing {path} ...")

        temp_path = cache_path.with_suffix(".tmp")
        chunks = iter(text_chunks(path))

        with open(temp_path, "wb") as f:
            while batch := list(islice(chunks, 32)):
                encoded = tokenizer(
                    batch,
                    add_special_tokens=False,
                )["input_ids"]

                for ids in encoded:
                    np.asarray(ids, dtype=dtype).tofile(f)

        temp_path.replace(cache_path)
        print(f"Saved: {cache_path}")
    else:
        print(f"Loading cached tokens: {cache_path}")

    return np.memmap(cache_path, dtype=dtype, mode="r")


class LLMDataset(Dataset):
    def __init__(self, path, block_size, tokenizer):
        self.data = load_tokens(path, tokenizer)
        self.block_size = block_size

        assert len(self.data) > block_size

    def __len__(self):
        return (len(self.data) - 1) // self.block_size

    def __getitem__(self, idx):
        start = idx * self.block_size

        window = np.asarray(
            self.data[start:start + self.block_size + 1],
            dtype=np.int64,
        )

        tokens = torch.from_numpy(window)

        return tokens[:-1], tokens[1:]


def create_dataloaders(
    train_path,
    val_path,
    tokenizer,
    block_size=256,
    batch_size=32,
    num_workers=0,
):
    train_dataset = LLMDataset(
        train_path, block_size, tokenizer
    )
    val_dataset = LLMDataset(
        val_path, block_size, tokenizer
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader
