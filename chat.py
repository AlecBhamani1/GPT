"""chat.py — talk to the model interactively.

If you point it at a chat-fine-tuned checkpoint (gpt_chat.pt) it holds a real
back-and-forth conversation: it remembers the dialogue, formats each turn with
the chat template, streams the reply token-by-token, and stops at end-of-turn.

If you point it at a raw pretrained base (gpt_model.pt) it falls back to plain
text continuation.

    python chat.py                 # loads gpt_chat.pt if present, else gpt_model.pt
    python chat.py --ckpt foo.pt   # a specific checkpoint
    python chat.py --base          # force plain continuation mode

Commands inside the chat:  /reset clears history   /quit exits
"""
import os
import sys
import argparse

import torch
import tiktoken

from model import GPT, pick_device, pick_dtype, amp_ctx
from chat_format import encode_prompt, SYSTEM_PREFIX, EOT, STOP_TEXT


def pick_checkpoint(explicit):
    if explicit:
        return explicit
    for cand in ('gpt_chat.pt', 'gpt_model.pt'):
        if os.path.exists(cand):
            return cand
    raise SystemExit("no checkpoint found -- train first (python train.py) "
                     "then optionally fine-tune (python finetune_chat.py).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--base', action='store_true', help='force plain continuation mode')
    ap.add_argument('--max-new', type=int, default=256)
    ap.add_argument('--temperature', type=float, default=0.8)
    ap.add_argument('--top-k', type=int, default=200)
    ap.add_argument('--top-p', type=float, default=0.95)
    ap.add_argument('--repetition-penalty', type=float, default=1.15)
    ap.add_argument('--prompt', default=None,
                    help='answer this one message and exit (non-interactive)')
    ap.add_argument('--system', default=None,
                    help='override the system/persona preamble (chat mode)')
    args = ap.parse_args()

    # Fail fast on nonsensical sampling settings rather than producing junk.
    if args.temperature <= 0:
        raise SystemExit("--temperature must be > 0")
    if not (0 < args.top_p <= 1):
        raise SystemExit("--top-p must be in (0, 1]")
    if args.top_k is not None and args.top_k < 1:
        raise SystemExit("--top-k must be >= 1")
    if args.max_new < 1:
        raise SystemExit("--max-new must be >= 1")
    if args.repetition_penalty <= 0:
        raise SystemExit("--repetition-penalty must be > 0")

    device = pick_device()
    dtype = pick_dtype(device)
    path = pick_checkpoint(args.ckpt)

    model, ckpt = GPT.from_checkpoint(path, map_location=device)
    model.to(device).eval()
    chat_mode = ckpt.get('finetuned', False) and not args.base

    enc = tiktoken.get_encoding(ckpt.get('tokenizer', 'gpt2'))
    decode = enc.decode

    kind = "chat" if chat_mode else "base (continuation)"
    vl = f", val {ckpt['val_loss']:.3f}" if 'val_loss' in ckpt else ""
    print(f"Loaded {path} ({model.num_params()/1e6:.1f}M params, {kind} mode, "
          f"{ckpt.get('iter','?')} steps{vl}) on {device}.")
    if not args.prompt:
        print("Type a message. /reset clears history, /quit exits.\n")

    # custom persona preamble, or the default the model was fine-tuned with
    system_prompt = (args.system.rstrip() + "\n") if args.system else SYSTEM_PREFIX
    history = []                                     # list of (role, content)

    while True:
        if args.prompt is not None:
            user = args.prompt.strip()               # one-shot mode
        else:
            try:
                user = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
        if user in ('/quit', 'quit', 'exit', '/exit'):
            break
        if user == '/reset' and args.prompt is None:
            history = []
            print("(history cleared)\n")
            continue
        if not user:
            if args.prompt is not None:
                break
            continue

        if chat_mode:
            # Sliding window: drop the oldest turns until the prompt + room for a
            # full reply fits the context, so long conversations keep working.
            budget = model.cfg.block_size - args.max_new
            turns = history + [('user', user)]
            ids = encode_prompt(turns, system=system_prompt)
            while len(ids) > budget and len(turns) > 1:
                turns = turns[2:]                      # drop an oldest user+assistant pair
                ids = encode_prompt(turns, system=system_prompt)
            ids = ids[-budget:]                        # final guard if a single turn is huge
        else:
            ids = enc.encode_ordinary(user)[-model.cfg.block_size:]
        idx = torch.tensor([ids], dtype=torch.long, device=device)

        # Stream the reply: decode the running token buffer and print only the
        # newly-revealed text (BPE tokens can be partial bytes, so we diff full
        # decodes rather than decoding each token alone). Stop at a "\nUser:".
        toks, shown = [], ""
        sys.stdout.write("bot> ")
        sys.stdout.flush()

        def on_token(tok):
            nonlocal shown
            toks.append(tok)
            text = decode(toks)
            if STOP_TEXT in text:                      # ran into a fake user turn
                return
            # hold back a few trailing chars in case they're the start of STOP_TEXT
            safe = text[:-len(STOP_TEXT)] if len(text) > len(STOP_TEXT) else ""
            if len(safe) > len(shown):
                sys.stdout.write(safe[len(shown):])
                sys.stdout.flush()
                shown = safe

        with amp_ctx(device, dtype):
            out = model.generate(
                idx, max_new_tokens=args.max_new,
                temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                eos_token_id=EOT if chat_mode else None,
                on_token=on_token,
            )

        reply_ids = out[0].tolist()[len(ids):]
        text = decode(reply_ids).replace(decode([EOT]), '')
        text = text.split(STOP_TEXT)[0].strip()        # trim any fallback user turn
        sys.stdout.write(text[len(shown):] + "\n\n")   # flush whatever was held back
        sys.stdout.flush()

        if chat_mode:
            history.append(('user', user))
            history.append(('assistant', text))

        if args.prompt is not None:
            break                                       # one-shot: done


if __name__ == '__main__':
    main()
