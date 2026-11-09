import torch
import torch.nn as nn
import torch.nn.functional as F

def causal_attention(q, k, v):
  return F.scaled_dot_product_attention(q, k, v, is_causal=True)

class CausalSelfAttention(nn.Module):
  def __init__(self, c):
    super().__init__()
    assert c.n_embd % c.n_head == 0
    self.n_head = c.n_head
    self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd)
    self.proj = nn.Linear(c.n_embd, c.n_embd)
    self.drop = nn.Dropout(c.dropout)

  def forward(self, x):
    B, T, C = x.shape
    q, k, v = self.qkv(x).split(C, dim=2) # B, T, 3*C -> (B,T,C)
    q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1,2) for t in (q,k,v))
    y = causal_attention(q, k, v) # B, n_head, T, n_embd
    y = y.transpose(1, 2).contiguous().view(B,T,C)
    return self.drop(self.proj(y))

class Block(nn.Module):
  def __init__(self, c):
    super().__init__()
    self.ln1 = nn.LayerNorm(c.n_embd)
    self.attn = CausalSelfAttention(c)
    self.ln2 = nn.LayerNorm(c.n_embd)
    self.mlp = nn.Sequential(
        nn.Linear(c.n_embd, 4 * c.n_embd),
        nn.GELU(),
        nn.Linear(4 * c.n_embd, c.n_embd),
        nn.Dropout(c.dropout)
    )

  def forward(self, x):
    # x -> 대충 B,T,C -> 마지막 차원을 대략 분산 1로 고정시키기
    x = x + self.attn(self.ln1(x))
    # B, T, C -> B,T,C 
    x = x + self.mlp(self.ln2(x))
    return x

class GPT(nn.Module):
  def __init__(self, c):
    super().__init__()
    self.c = c
    self.tok_emb = nn.Embedding(c.vocab_size, c.n_embd)
    self.pos_emb = nn.Embedding(c.block_size, c.n_embd)
    self.drop = nn.Dropout(c.dropout)
    self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
    self.ln_f = nn.LayerNorm(c.n_embd)
    self.head = nn.Linear(c.n_embd, c.vocab_size)

    self.apply(self._init) 

  def _init(self, m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, std=0.02)

  def forward(self, idx, targets=None):
    B, T = idx.shape
    assert T <= self.c.block_size
    pos = torch.arange(T, device=idx.device)
    tok_emb = self.tok_emb(idx)
    pos_emb = self.pos_emb(pos)
    x = self.drop(tok_emb + pos_emb)
    for blk in self.blocks:
      x = blk(x)
    logits = self.head(self.ln_f(x))
    # B, T, n_embd ->
    # B, T
    # cross_entropy (개수, 정답) (개수, 로짓)
    loss = None
    if targets is not None:
      loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    return logits, loss
