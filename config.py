from dataclasses import dataclass, field
import sysconfig


@dataclass
class Config:
    vocab_size: int
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1


# Explicit name for new code; Config remains available for the notebook.
ModelConfig = Config


@dataclass
class DataConfig:
    corpus_dir: str = field(default_factory=lambda: sysconfig.get_paths()["stdlib"])
    batcher: str = "random"
    tokenizer: str = "char"
    tokenizer_name: str = "gpt2"
    tokenizer_revision: str | None = None
    train_fraction: float = 0.9
    split_seed: int = 0


@dataclass
class TrainConfig:
    max_iters: int = 4000
    batch_size: int = 64
    eval_every: int = 250
    eval_iters: int = 50
    save_every: int = 250
    lr: float = 6e-4
    min_lr: float = 6e-5
    warmup: int = 100
    grad_clip: float = 1.0
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    seed: int = 1337
