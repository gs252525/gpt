
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

benchmark(dtype=torch.float32)
benchmark(dtype=torch.float16)
