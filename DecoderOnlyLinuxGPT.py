#Originally hand-written following Andrej Karpathy's "Let's build GPT from scratch".
#youtube link: https://www.youtube.com/watch?v=kCc8FmEb1nY&t=2799s
#
#Now scaled up for GENERAL WEB-TEXT pretraining at GPT-2-small size (~124M params)
#on a single DGX Spark:
#   * GPT-2 BPE tokenizer (tiktoken)  -- this IS byte-level BPE
#   * data is memory-mapped from train.bin / val.bin  (run prepare_data.py FIRST)
#   * bf16 autocast, cosine LR schedule + warmup, grad clipping, weight decay
#
#This file is the teaching version of nanoGPT (https://github.com/karpathy/nanoGPT).
#If you hit throughput or stability walls, nanoGPT is the validated reference for
#exactly this setup (it batches the attention heads and uses flash attention).
import os
import math
import time
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

# ----------------------- config: GPT-2 small (~124M) -----------------------
# These are starting points tuned for general web text on one DGX Spark. The
# Spark's 128GB unified memory means you can raise batch_size; its memory
# bandwidth is the real limit, so training is throughput-bound, not memory-bound.
batch_size       = 12       #sequences per micro-step (raise on the Spark; 128GB is plenty)
block_size       = 1024     #context length in TOKENS (not characters anymore)
grad_accum_steps = 40       #effective batch ~ 12*1024*40 ~= 0.5M tokens/step
max_iters        = 100000   #optimizer steps for a LONG run (~5 epochs over 10BT)
warmup_iters     = 2000
lr_decay_iters   = 100000   #usually == max_iters
eval_interval    = 500      #eval + checkpoint cadence
eval_iters       = 100
learning_rate    = 6e-4     #peak LR (GPT-2 small)
min_lr           = 6e-5     #final LR after cosine decay
weight_decay     = 0.1
beta1, beta2     = 0.9, 0.95
grad_clip        = 1.0
n_embd           = 768
n_head           = 12
n_layer          = 12
dropout          = 0.0      #0 for large-data pretraining (no overfitting risk)
compile_model    = True     #torch.compile -> big speedup on Blackwell (set False to debug)
resume           = True     #resume from gpt_model.pt if it exists (safe for long runs)
device           = 'cuda' if torch.cuda.is_available() else 'cpu'
device_type      = 'cuda' if 'cuda' in device else 'cpu'
#Precision: bf16 is the right call for TRAINING on the Spark's Blackwell GPU --
#wide dynamic range, no loss-scaling needed, fully accelerated. fp16 would be a
#downgrade (needs a GradScaler, more fragile). fp8 (via NVIDIA Transformer Engine)
#is the only genuinely faster path, but adds complexity/risk -- revisit later.
dtype            = torch.bfloat16

#GPT-2 byte-level BPE. vocab is 50257, padded up to 50304 (a multiple of 64) so
#the embedding/output matmuls are GPU-friendly. The extra rows are never emitted.
enc        = tiktoken.get_encoding('gpt2')
vocab_size = 50304
encode     = lambda s: enc.encode_ordinary(s)   #string -> list[int]
decode     = lambda l: enc.decode(l)            #list[int] -> string


# ----------------------- data: memory-mapped token files -----------------------
def get_batch(split):
    #Re-open the memmap every call: keeping one around leaks memory over a long run.
    path = 'train.bin' if split == 'train' else 'val.bin'
    data = np.memmap(path, dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        #pin + non-blocking transfer overlaps the copy with compute
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with torch.autocast(device_type=device_type, dtype=dtype):
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(it):
    #linear warmup, then cosine decay down to min_lr
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# ----------------------- model -----------------------
class CausalSelfAttention(nn.Module):
    """All heads at once: one batched QKV projection + flash attention.

    Replaces the old per-head Head/MultiHeadAttention (which ran a separate set
    of matmuls per head and built the full T x T attention matrix explicitly).
    Here q, k, v for every head come from a single Linear, and
    F.scaled_dot_product_attention runs the fused/flash kernel with the causal
    mask applied implicitly via is_causal=True -- no tril buffer needed.
    """

    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        #one projection produces q, k and v for all heads (n_embd -> 3*n_embd)
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        #output projection back into the residual stream
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        #split the fused projection into q, k, v, each (B, T, C)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        #reshape each into (B, n_head, T, head_size) so heads are a batch dim
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        #flash attention; is_causal=True applies the lower-triangular mask
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        #(B, n_head, T, hs) -> (B, T, C): put the heads back together
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.gelu = nn.GELU()                   #GELU is what GPT-2 uses (was ReLU)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = CausalSelfAttention(n_embd, n_head)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)              #final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        #weight tying: share the token-embedding matrix with the output projection
        #(GPT-2 does this). At a 50k vocab this saves ~38M params and helps quality.
        self.lm_head.weight = self.token_embedding_table.weight
        self.apply(self._init_weights)

        #GPT-2 residual scaling: shrink the projections that write back into the
        #residual stream by 1/sqrt(2*n_layer) so its variance doesn't grow with depth.
        for name, p in self.named_parameters():
            if name.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)                       #(B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)                                                #<- was missing before
        logits = self.lm_head(x)                                        #(B,T,vocab_size)

        if targets is None:
            return logits, None
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    def configure_optimizers(self):
        #weight-decay the 2D matrices (Linear/embedding); never decay 1D params
        #(biases, LayerNorm). This is the standard GPT-2 recipe.
        decay, no_decay = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {'params': decay, 'weight_decay': weight_decay},
            {'params': no_decay, 'weight_decay': 0.0},
        ]
        return torch.optim.AdamW(groups, lr=learning_rate, betas=(beta1, beta2))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]               #crop to the context window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature        #last step, scaled
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ----------------------- train (only when run directly) -----------------------
#Skipped when chat.py imports the model classes above, so importing stays fast.
if __name__ == '__main__':
    if not (os.path.exists('train.bin') and os.path.exists('val.bin')):
        raise SystemExit("train.bin / val.bin not found -- run `python prepare_data.py` first.")

    torch.manual_seed(1337)
    torch.set_float32_matmul_precision('high')   #TF32 for fp32 matmuls (free speedup)

    model = BigramLanguageModel().to(device)
    raw_model = model                            #uncompiled handle, used for saving
    optimizer = raw_model.configure_optimizers()

    #resume from the last checkpoint if present, so a long run survives interruptions
    start_iter = 0
    if resume and os.path.exists('gpt_model.pt'):
        ckpt = torch.load('gpt_model.pt', map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt['model_state'])
        if 'optimizer_state' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state'])
        start_iter = ckpt.get('iter', 0)
        print(f"resumed from step {start_iter}")

    n_params = sum(p.numel() for p in raw_model.parameters())
    print(f"model: {n_params/1e6:.1f}M params | device={device} | dtype={dtype} | compile={compile_model}")

    if compile_model:
        model = torch.compile(model)             #first step is slow while it compiles

    tokens_per_step = batch_size * block_size * grad_accum_steps
    t_window = time.time()
    for it in range(start_iter, max_iters):
        #set the cosine-scheduled learning rate for this step
        lr = get_lr(it)
        for g in optimizer.param_groups:
            g['lr'] = lr

        #periodically evaluate, log throughput, and checkpoint (save often on long runs)
        if it % eval_interval == 0:
            losses = estimate_loss()
            if it > start_iter:
                tps = tokens_per_step * eval_interval / (time.time() - t_window)
                rate = f"{tps/1e3:.1f}k tok/s"
            else:
                rate = "warming up"
            print(f"step {it}: train {losses['train']:.4f} | val {losses['val']:.4f} | lr {lr:.2e} | {rate}")
            torch.save({
                'model_state': raw_model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'config': {'vocab_size': vocab_size, 'n_embd': n_embd, 'n_head': n_head,
                           'n_layer': n_layer, 'block_size': block_size, 'dropout': dropout},
                'tokenizer': 'gpt2',
                'iter': it,
            }, 'gpt_model.pt')
            t_window = time.time()               #don't count eval/save time in throughput

        #gradient accumulation: many micro-batches -> one large effective batch
        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum_steps):
            xb, yb = get_batch('train')
            with torch.autocast(device_type=device_type, dtype=dtype):
                logits, loss = model(xb, yb)
                loss = loss / grad_accum_steps   #scale so accumulated grads average
            loss.backward()
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)
        optimizer.step()

    print("done -> gpt_model.pt saved.  Chat with it:  python chat.py")
