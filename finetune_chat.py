"""finetune_chat.py — turn a pretrained base checkpoint into a chat assistant.

Supervised fine-tuning (SFT) on conversations. Each conversation is rendered with
the shared chat template; loss is computed ONLY on the assistant's replies (user
turns are masked out), so the model learns to *respond*, not to echo the prompt.

    python train.py                                  # 1. pretrain -> gpt_model.pt
    python finetune_chat.py --data sample_chat.jsonl # 2. fine-tune -> gpt_chat.pt
    python chat.py                                   # 3. chat (loads gpt_chat.pt)

Dataset format — one JSON object per line:
    {"messages": [{"role": "user", "content": "hi"},
                  {"role": "assistant", "content": "Hello! How can I help?"}]}
"""
import os
import json
import math
import time
import argparse
from dataclasses import replace

import numpy as np
import torch

from model import GPT, GPTConfig, pick_device, pick_dtype, amp_ctx, load_checkpoint
from chat_format import encode_example


def _validate_messages(messages, where):
    """Check one conversation's message list, with a clear error pointing at it."""
    if not isinstance(messages, list) or not messages:
        raise SystemExit(f"{where}: 'messages' must be a non-empty list.")
    saw_assistant = False
    for j, m in enumerate(messages):
        if not isinstance(m, dict) or 'role' not in m or 'content' not in m:
            raise SystemExit(f"{where}, message {j}: each message needs "
                             f"'role' and 'content'.")
        if m['role'] not in ('user', 'assistant'):
            raise SystemExit(f"{where}, message {j}: role must be 'user' or "
                             f"'assistant', got {m['role']!r}.")
        if not isinstance(m['content'], str) or not m['content'].strip():
            raise SystemExit(f"{where}, message {j}: 'content' must be non-empty text.")
        saw_assistant |= m['role'] == 'assistant'
    if not saw_assistant:
        # nothing to learn — every token would be masked out of the loss
        raise SystemExit(f"{where}: needs at least one 'assistant' reply to train on.")


def load_dataset(path):
    """Read a JSONL of conversations -> list of per-conversation (ids, mask).

    Validates each line with a friendly, line-numbered message so a typo in the
    data fails fast here instead of crashing deep in the training loop."""
    examples = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            where = f"{path}:{lineno}"
            try:
                conv = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{where}: invalid JSON ({e.msg}).")
            if not isinstance(conv, dict) or 'messages' not in conv:
                raise SystemExit(f"{where}: each line must be a JSON object with a "
                                 f"'messages' key.")
            _validate_messages(conv['messages'], where)
            examples.append(encode_example(conv['messages']))
    if not examples:
        raise SystemExit(f"{path} has no conversations.")
    return examples


def flatten(examples, block_size):
    """Concatenate examples into flat (ids, mask) arrays, padded to >= one block."""
    ids, mask = [], []
    for e_ids, e_mask in examples:
        ids.extend(e_ids)
        mask.extend(e_mask)
    if len(ids) <= block_size + 1:
        pad = block_size + 2 - len(ids)
        ids.extend([0] * pad)
        mask.extend([0] * pad)        # padding is masked out of the loss
    return np.array(ids, dtype=np.int64), np.array(mask, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='sample_chat.jsonl', help='JSONL of conversations')
    ap.add_argument('--base', default='gpt_model.pt', help='pretrained checkpoint')
    ap.add_argument('--out', default='gpt_chat.pt', help='where to save the chat model')
    ap.add_argument('--iters', type=int, default=int(os.environ.get('MAX_ITERS', 1000)))
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=3e-5)        # gentle: don't wreck the base
    ap.add_argument('--dropout', type=float, default=0.1)    # mild regularization for SFT
    ap.add_argument('--eval-interval', type=int, default=100)
    ap.add_argument('--val-fraction', type=float, default=0.15,
                    help='held-out conversations for picking the best checkpoint')
    ap.add_argument('--patience', type=int, default=5,
                    help='stop early after this many evals without val improvement')
    args = ap.parse_args()

    if not os.path.exists(args.base):
        raise SystemExit(f"base checkpoint {args.base} not found -- run `python train.py` first.")
    if not os.path.exists(args.data):
        raise SystemExit(f"dataset {args.data} not found.")

    device = pick_device()
    dtype = pick_dtype(device)
    torch.manual_seed(1337)

    # Load the base, then rebuild with fine-tuning dropout (dropout has no params,
    # so the pretrained state_dict still loads cleanly).
    ckpt = load_checkpoint(args.base, map_location=device)
    cfg = replace(GPTConfig(**ckpt['config']), dropout=args.dropout)
    model = GPT(cfg)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)

    examples = load_dataset(args.data)
    block_size = cfg.block_size

    # Hold out whole conversations for validation (only if there are enough to
    # spare). With too few, validate on the training stream so we still pick a
    # sensible best checkpoint rather than guessing.
    n_val = int(len(examples) * args.val_fraction)
    if len(examples) >= 4 and n_val >= 1:
        rng = np.random.default_rng(1337)
        perm = rng.permutation(len(examples))
        val_ex = [examples[i] for i in perm[:n_val]]
        train_ex = [examples[i] for i in perm[n_val:]]
        have_val = True
    else:
        train_ex, val_ex, have_val = examples, examples, False

    train_ids, train_mask = flatten(train_ex, block_size)
    val_ids, val_mask = flatten(val_ex, block_size)
    print(f"loaded {len(examples)} conversations "
          f"({len(train_ex)} train / {len(val_ex) if have_val else 0} val) | "
          f"{int(train_mask.sum()):,} trained tokens | device={device}")

    def get_batch(ids, mask):
        ix = torch.randint(len(ids) - block_size - 1, (args.batch_size,))
        x = torch.stack([torch.from_numpy(ids[i:i + block_size]) for i in ix])
        y = torch.stack([torch.from_numpy(ids[i + 1:i + 1 + block_size]) for i in ix])
        m = torch.stack([torch.from_numpy(mask[i + 1:i + 1 + block_size]) for i in ix])
        y = y.masked_fill(m == 0, -1)            # ignore non-assistant tokens in the loss
        return x.to(device), y.to(device)

    @torch.no_grad()
    def val_loss(n_batches=10):
        model.eval()
        losses = []
        for _ in range(n_batches):
            xb, yb = get_batch(val_ids, val_mask)
            with amp_ctx(device, dtype):
                _, l = model(xb, yb)
            losses.append(l.item())
        model.train()
        return sum(losses) / len(losses)

    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=args.lr,
                                           betas=(0.9, 0.95))
    model.train()
    t0 = time.time()
    best_val, since_best = float('inf'), 0
    saved = False
    for it in range(args.iters):
        # cosine decay with a short warmup
        warm = max(10, args.iters // 20)
        if it < warm:
            lr = args.lr * (it + 1) / warm
        else:
            r = (it - warm) / max(1, args.iters - warm)
            lr = 0.1 * args.lr + 0.9 * args.lr * 0.5 * (1 + math.cos(math.pi * r))
        for g in optimizer.param_groups:
            g['lr'] = lr

        xb, yb = get_batch(train_ids, train_mask)
        with amp_ctx(device, dtype):
            _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if it % args.eval_interval == 0 or it == args.iters - 1:
            vl = val_loss()
            tag = ""
            # Save the BEST checkpoint by val loss, not just the last step.
            if vl < best_val:
                best_val, since_best = vl, 0
                model.save(args.out, tokenizer='gpt2', iter=it, finetuned=True,
                           val_loss=vl)
                saved, tag = True, "  <- best, saved"
            else:
                since_best += 1
            print(f"step {it}: train {loss.item():.4f} | val {vl:.4f} | "
                  f"lr {lr:.2e} | {time.time()-t0:.1f}s{tag}")
            if have_val and since_best >= args.patience:
                print(f"early stop: no val improvement for {args.patience} evals")
                break

    if not saved:        # safety net (e.g. iters < eval_interval)
        model.save(args.out, tokenizer='gpt2', iter=args.iters, finetuned=True,
                   val_loss=best_val)
    print(f"done -> {args.out} saved (best val {best_val:.4f}).  Chat:  python chat.py")


if __name__ == '__main__':
    main()
