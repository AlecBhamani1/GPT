"""eval.py — measure model quality with perplexity (lower = better).

A concrete number to answer "is the model getting better?" Runs inference only —
no training. Two modes:

    python eval.py --ckpt gpt_model.pt --bin val.bin           # pretraining ppl
    python eval.py --ckpt gpt_chat.pt --chat sample_chat.jsonl # assistant-reply ppl

The chat mode scores perplexity over ONLY the assistant tokens (the same loss mask
fine-tuning uses), so it reflects how well the model produces replies, not prompts.
Coverage is deterministic (contiguous blocks), so runs are comparable.
"""
import os
import math
import argparse

import numpy as np
import torch

from model import GPT, pick_device, pick_dtype, amp_ctx
from finetune_chat import load_dataset, flatten


def iter_blocks(ids, mask, block_size, batch_size):
    """Yield (x, y, m) numpy batches over contiguous, non-overlapping blocks."""
    n_blocks = (len(ids) - 1) // block_size
    for start in range(0, n_blocks, batch_size):
        xs, ys, ms = [], [], []
        for b in range(start, min(start + batch_size, n_blocks)):
            i = b * block_size
            xs.append(ids[i:i + block_size])
            ys.append(ids[i + 1:i + 1 + block_size])
            ms.append(mask[i + 1:i + 1 + block_size])
        yield np.array(xs), np.array(ys), np.array(ms)


@torch.no_grad()
def perplexity(model, ids, mask, block_size, batch_size, device, dtype):
    """Token-weighted perplexity over the counted (mask==1) tokens."""
    model.eval()
    total_nll, total_tok = 0.0, 0
    for xb, yb, mb in iter_blocks(ids, mask, block_size, batch_size):
        x = torch.from_numpy(xb).long().to(device)
        y = torch.from_numpy(np.where(mb == 0, -1, yb)).long().to(device)
        n = int((y != -1).sum().item())
        if n == 0:
            continue
        with amp_ctx(device, dtype):
            _, loss = model(x, y)            # mean NLL over counted tokens in the batch
        total_nll += loss.item() * n         # un-average so batches combine correctly
        total_tok += n
    if total_tok == 0:
        raise SystemExit("no tokens to score (empty / fully-masked data).")
    return math.exp(total_nll / total_tok), total_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--bin', help='a tokenized .bin (e.g. val.bin) for raw LM perplexity')
    ap.add_argument('--chat', help='a chat JSONL for assistant-reply perplexity')
    ap.add_argument('--batch-size', type=int, default=8)
    args = ap.parse_args()
    if bool(args.bin) == bool(args.chat):
        raise SystemExit("pass exactly one of --bin or --chat")
    if not os.path.exists(args.ckpt):
        raise SystemExit(f"{args.ckpt} not found")

    device = pick_device()
    dtype = pick_dtype(device)
    model, _ = GPT.from_checkpoint(args.ckpt, map_location=device)
    model.to(device)
    bs = model.cfg.block_size

    if args.bin:
        data = np.memmap(args.bin, dtype=np.uint16, mode='r')
        ids = np.asarray(data, dtype=np.int64)
        mask = np.ones_like(ids)                         # score every token
        label = args.bin
    else:
        ids, mask = flatten(load_dataset(args.chat), bs)
        label = f"{args.chat} (assistant tokens only)"

    ppl, n = perplexity(model, ids, mask, bs, args.batch_size, device, dtype)
    print(f"{label}: perplexity {ppl:.2f} over {n:,} tokens "
          f"({model.num_params()/1e6:.1f}M params, {device})")


if __name__ == '__main__':
    main()
