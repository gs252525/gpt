import psutil
import os 

def print_memory():
    process = psutil.Process(os.getpid())
    ram = process.memory_info().rss / 1024**3
    print(f"RAM 사용량: {ram:.2f} GB")

def print_model_size(model): 
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Model size: {total_params / 1e6:.2f}M")

    learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) 

    print(f"learnable parameters: {learnable_params:,}")
    print(f"Gradient size: {learnable_params / 1e6:.2f}M")
