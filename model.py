"""model.py — the GPT architecture, config, and (KV-cached) generation.

Importable with no side effects: training/chat scripts import `GPT`, `GPTConfig`,
and the device/dtype helpers from here. Originally hand-written following Andrej
Karpathy's "Let's build GPT from scratch", now a clean GPT-2 with the pieces a
chatbot needs: a KV cache for fast incremental decoding, plus top-k / top-p /
repetition-penalty sampling and an EOS stop condition.
"""
import math
from contextlib import nullcontext
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn import functional as F


# ----------------------- device / precision helpers -----------------------
def pick_device():
    """Best available accelerator: CUDA > MPS (Apple GPU) > CPU."""
    if torch.cuda.is_available():
        return 'cuda'
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def pick_dtype(device):
    """bf16 on CUDA (wide range, no loss scaling); fp32 elsewhere (MPS/CPU are
    correct and fast enough in fp32, and dodge flaky half-precision autocast)."""
    return torch.bfloat16 if device == 'cuda' else torch.float32


def amp_ctx(device, dtype):
    """autocast only where it actually helps; a no-op context otherwise."""
    if device == 'cuda':
        return torch.autocast(device_type='cuda', dtype=dtype)
    if device == 'cpu' and dtype == torch.bfloat16:
        return torch.autocast(device_type='cpu', dtype=dtype)
    return nullcontext()


# ----------------------- config -----------------------
@dataclass
class GPTConfig:
    vocab_size: int = 50304   # GPT-2 BPE (50257) padded to a multiple of 64
    block_size: int = 1024    # context length in tokens
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0      # 0 for pretraining; raise (~0.1) for fine-tuning

    def n_params(self):
        # quick analytic estimate (embeddings counted once thanks to weight tying)
        return (self.vocab_size * self.n_embd
                + self.block_size * self.n_embd
                + self.n_layer * (12 * self.n_embd ** 2 + 13 * self.n_embd)
                + 2 * self.n_embd)


# Named presets so you don't need a DGX Spark to train *something*.
#   tiny  — runs on a laptop CPU/MPS in minutes (smoke tests, demos)
#   small — a single consumer GPU can pretrain this overnight
#   gpt2  — the original 124M target for the Spark
PRESETS = {
    'tiny':  GPTConfig(block_size=256, n_layer=4, n_head=4, n_embd=128, dropout=0.0),
    'small': GPTConfig(block_size=512, n_layer=8, n_head=8, n_embd=512, dropout=0.0),
    'gpt2':  GPTConfig(block_size=1024, n_layer=12, n_head=12, n_embd=768, dropout=0.0),
}


# ----------------------- model -----------------------
class CausalSelfAttention(nn.Module):
    """All heads at once: one fused QKV projection + flash attention, with an
    optional KV cache so incremental decoding is O(T) per token instead of O(T²)."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.dropout = cfg.dropout
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, past_kv=None, use_cache=False):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat((past_k, k), dim=2)   # extend the cached keys/values
            v = torch.cat((past_v, v), dim=2)
        present = (k, v) if use_cache else None

        # Prefill (T>1, no cache) needs the causal mask. A single cached decode
        # step (T==1) attends to ALL cached keys, so no mask is needed there.
        is_causal = past_kv is None and T > 1
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, present


class FeedForward(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.sa = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.ffwd = FeedForward(cfg)

    def forward(self, x, past_kv=None, use_cache=False):
        attn, present = self.sa(self.ln1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn
        x = x + self.ffwd(self.ln2(x))
        return x, present


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.use_checkpoint = False    # set True to trade compute for memory in training
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # weight tying: share the token embedding with the output projection (GPT-2)
        self.lm_head.weight = self.token_embedding_table.weight
        self.apply(self._init_weights)

        # GPT-2 residual scaling: shrink projections that write to the residual
        # stream by 1/sqrt(2*n_layer) so variance doesn't grow with depth.
        for name, p in self.named_parameters():
            if name.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        # subtract the position table; token table is tied to lm_head (counted once)
        return sum(p.numel() for p in self.parameters())

    def _transformer(self, idx, past_kvs=None, start_pos=0, use_cache=False):
        """Shared trunk: embeddings -> blocks -> final LN. Returns (x, presents)."""
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos = torch.arange(start_pos, start_pos + T, device=idx.device)
        x = self.drop(tok_emb + self.position_embedding_table(pos))
        presents = [] if use_cache else None
        ckpt = self.use_checkpoint and self.training and not use_cache
        for i, block in enumerate(self.blocks):
            past = past_kvs[i] if past_kvs is not None else None
            if ckpt:
                # recompute activations in backward instead of storing them:
                # ~sqrt(n_layer) less activation memory, lets you train bigger.
                x, present = torch.utils.checkpoint.checkpoint(
                    block, x, past, use_cache, use_reentrant=False)
            else:
                x, present = block(x, past_kv=past, use_cache=use_cache)
            if use_cache:
                presents.append(present)
        return self.ln_f(x), presents

    def forward(self, idx, targets=None):
        """Standard training/eval interface: returns (logits, loss). Tokens in
        `targets` set to -1 are ignored by the loss (used to mask user turns)."""
        x, _ = self._transformer(idx)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
        )
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        # weight-decay the 2D matrices (Linear/embedding); never decay 1D params
        # (biases, LayerNorm) — the standard GPT-2 recipe.
        decay, no_decay = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {'params': decay, 'weight_decay': weight_decay},
            {'params': no_decay, 'weight_decay': 0.0},
        ]
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None,
                 repetition_penalty=1.0, eos_token_id=None, on_token=None):
        """Autoregressively sample continuations with a KV cache.

        temperature/top_k/top_p shape the distribution; repetition_penalty (>1)
        discourages repeating tokens already in the context; generation stops
        early at eos_token_id. `on_token(int)` (if given) is called per new token
        for streaming. Returns the full sequence (prompt + generated)."""
        self.eval()
        block_size = self.cfg.block_size
        idx = idx[:, -block_size:]                 # never start beyond the window
        generated = idx
        past_kvs, start_pos = None, 0
        cur = idx

        for _ in range(max_new_tokens):
            x, past_kvs = self._transformer(cur, past_kvs=past_kvs,
                                            start_pos=start_pos, use_cache=True)
            start_pos += cur.size(1)
            logits = self.lm_head(x[:, -1, :])     # only the last position matters

            if repetition_penalty != 1.0:
                for tok in set(generated[0].tolist()):
                    logits[0, tok] /= repetition_penalty

            logits = logits / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            if top_p is not None:
                logits = _top_p_filter(logits, top_p)

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, idx_next), dim=1)
            if on_token is not None:
                on_token(idx_next.item())
            if eos_token_id is not None and idx_next.item() == eos_token_id:
                break
            if start_pos + 1 >= block_size:        # cache full — stop cleanly
                break
            cur = idx_next                          # feed only the new token next
        return generated

    # ---- checkpoint helpers ----
    def save(self, path, **extra):
        torch.save({'model_state': self.state_dict(),
                    'config': asdict(self.cfg), **extra}, path)

    @classmethod
    def from_checkpoint(cls, ckpt, map_location='cpu'):
        if isinstance(ckpt, str):
            ckpt = torch.load(ckpt, map_location=map_location, weights_only=False)
        cfg = GPTConfig(**ckpt['config'])
        model = cls(cfg)
        model.load_state_dict(ckpt['model_state'])
        return model, ckpt


def _top_p_filter(logits, top_p):
    """Nucleus filtering: keep the smallest set of tokens whose cumulative
    probability exceeds top_p, mask the rest."""
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    remove = cum > top_p
    remove[..., 1:] = remove[..., :-1].clone()    # keep at least the top token
    remove[..., 0] = False
    logits = logits.clone()
    logits.scatter_(-1, sorted_idx, sorted_logits.masked_fill(remove, -float('inf')))
    return logits


# Backwards-compatible alias: the model used to be (mis)named BigramLanguageModel.
BigramLanguageModel = GPT
