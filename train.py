"""train.py — pretrain the GPT on next-token prediction over a token stream.

This is the *pretraining* stage: the model learns general language from
`train.bin` / `val.bin` (build them with `prepare_data.py`). To make it chat,
fine-tune the resulting checkpoint with `finetune_chat.py`.

    python prepare_data.py                  # build train.bin / val.bin
    python train.py                         # pretrain (config from $GPT_CONFIG, default gpt2)
    GPT_CONFIG=small python train.py        # smaller, single-GPU-friendly
    GPT_CONFIG=tiny  python train.py        # laptop CPU/MPS smoke test

Long runs are resumable: checkpoints (model + optimizer + step) are written to
gpt_model.pt every eval_interval steps; rerun the same command to continue.
"""
import os
import math
import time

import numpy as np
import torch

from model import GPT, PRESETS, pick_device, pick_dtype, amp_ctx

# ----------------------- config -----------------------
CONFIG_NAME = os.environ.get('GPT_CONFIG', 'gpt2')
if CONFIG_NAME not in PRESETS:
    raise SystemExit(f"GPT_CONFIG='{CONFIG_NAME}' unknown -- pick one of "
                     f"{', '.join(PRESETS)}")
cfg = PRESETS[CONFIG_NAME]

# Training hyperparameters scale with the preset so smaller configs actually
# finish on modest hardware. Override any of these via env vars if you like.
_DEFAULTS = {
    'gpt2':  dict(batch_size=12, grad_accum_steps=40, max_iters=100_000, eval_interval=500),
    'small': dict(batch_size=24, grad_accum_steps=8,  max_iters=50_000,  eval_interval=500),
    'tiny':  dict(batch_size=16, grad_accum_steps=1,  max_iters=2_000,   eval_interval=200),
}[CONFIG_NAME]

batch_size       = int(os.environ.get('BATCH_SIZE', _DEFAULTS['batch_size']))
grad_accum_steps = int(os.environ.get('GRAD_ACCUM', _DEFAULTS['grad_accum_steps']))
max_iters        = int(os.environ.get('MAX_ITERS', _DEFAULTS['max_iters']))
eval_interval    = _DEFAULTS['eval_interval']
block_size       = cfg.block_size
warmup_iters     = max(100, max_iters // 50)
lr_decay_iters   = max_iters
eval_iters       = 50
learning_rate    = 6e-4
min_lr           = 6e-5
weight_decay     = 0.1
betas            = (0.9, 0.95)
grad_clip        = 1.0
compile_model    = os.environ.get('COMPILE', '1') == '1' and CONFIG_NAME != 'tiny'
resume           = True
out_path         = os.environ.get('CKPT', 'gpt_model.pt')

device      = pick_device()
device_type = device
dtype       = pick_dtype(device)


# ----------------------- data -----------------------
def get_batch(split):
    # Re-open the memmap each call: holding one open leaks memory over a long run.
    path = 'train.bin' if split == 'train' else 'val.bin'
    data = np.memmap(path, dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    if device == 'cuda':
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
            with amp_ctx(device_type, dtype):
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# ----------------------- train -----------------------
if __name__ == '__main__':
    if not (os.path.exists('train.bin') and os.path.exists('val.bin')):
        raise SystemExit("train.bin / val.bin not found -- run `python prepare_data.py` first.")

    torch.manual_seed(1337)
    torch.set_float32_matmul_precision('high')   # TF32 for fp32 matmuls (free on CUDA)

    model = GPT(cfg).to(device)
    raw_model = model
    # GRAD_CKPT=1 recomputes activations in the backward pass: far less memory,
    # so you can fit a bigger model / longer context on a small GPU (~20-30% slower).
    if os.environ.get('GRAD_CKPT', '0') == '1':
        raw_model.use_checkpoint = True
        print("gradient checkpointing ON")
    optimizer = raw_model.configure_optimizers(weight_decay, learning_rate, betas)

    start_iter = 0
    if resume and os.path.exists(out_path):
        ckpt = torch.load(out_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt['model_state'])
        if 'optimizer_state' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state'])
        start_iter = ckpt.get('iter', 0)
        print(f"resumed from step {start_iter}")

    print(f"config={CONFIG_NAME} | {raw_model.num_params()/1e6:.1f}M params | "
          f"device={device} | dtype={dtype} | compile={compile_model}")

    if compile_model:
        model = torch.compile(model)

    tokens_per_step = batch_size * block_size * grad_accum_steps
    t_window = time.time()
    for it in range(start_iter, max_iters):
        lr = get_lr(it)
        for g in optimizer.param_groups:
            g['lr'] = lr

        if it % eval_interval == 0:
            losses = estimate_loss()
            if it > start_iter:
                tps = tokens_per_step * eval_interval / (time.time() - t_window)
                rate = f"{tps/1e3:.1f}k tok/s"
            else:
                rate = "warming up"
            print(f"step {it}: train {losses['train']:.4f} | val {losses['val']:.4f} | "
                  f"lr {lr:.2e} | {rate}")
            raw_model.save(out_path, optimizer_state=optimizer.state_dict(),
                           tokenizer='gpt2', iter=it)
            t_window = time.time()

        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum_steps):
            xb, yb = get_batch('train')
            with amp_ctx(device_type, dtype):
                _, loss = model(xb, yb)
                loss = loss / grad_accum_steps
            loss.backward()
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)
        optimizer.step()

    raw_model.save(out_path, optimizer_state=optimizer.state_dict(), tokenizer='gpt2', iter=max_iters)
    print(f"done -> {out_path} saved.  Fine-tune for chat:  python finetune_chat.py")
