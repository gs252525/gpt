from dataclasses import dataclass

@dataclass
class Config:
  vocab_size: int
  block_size: int = 256
  n_layer: int = 6
  n_head: int = 6
  n_embd: int = 384
  dropout: float = 0.1