
import torch
import time

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("Device:", device)

def sync():
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()

def benchmark(n=4096, dtype=torch.float32):
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)

    for _ in range(10):
        c = a @ b
    sync()

    start = time.perf_counter()
    iters = 30

    for _ in range(iters):
        c = a @ b

    sync()
    elapsed = (time.perf_counter() - start) / iters

    flops = 2 * n**3
    tflops = flops / elapsed / 1e12

    print(f"Precision: {dtype}")
    print(f"Time: {elapsed*1000:.2f} ms")
    print(f"Measured: {tflops:.2f} TFLOPS")

from datasets import load_dataset

import sys

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


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

train_data = load_dataset(
    TRAIN_DATASET,
    split="train",
    streaming=True,
).shuffle(seed=SEED, buffer_size=1_000)

