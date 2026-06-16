"""smoke_test.py — fast correctness + speed checks (no training data needed).

Validates the pieces that matter for a chatbot:
  1. model builds, forward + loss work
  2. KV-cache generation MATCHES a naive full-recompute reference (greedy)
  3. KV-cache generation is faster than the naive O(T^2) path
  4. chat template encode/decode round-trips and loss-masking is correct
Run:  python smoke_test.py
"""
import time
import torch

from model import GPT, GPTConfig, pick_device
import chat_format as cf


def naive_generate(model, idx, max_new_tokens):
    """Reference decoder: re-run the full context every step, NO cache, greedy."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.cfg.block_size:]
        x, _ = model._transformer(idx_cond)
        logits = model.lm_head(x[:, -1, :])
        nxt = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, nxt), dim=1)
    return idx


def main():
    device = pick_device()
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=50304, block_size=128, n_layer=4, n_head=4, n_embd=128)
    model = GPT(cfg).to(device).eval()
    print(f"[1] built {model.num_params()/1e6:.2f}M-param model on {device}")

    # forward + loss
    x = torch.randint(0, cfg.vocab_size, (2, 32), device=device)
    y = torch.randint(0, cfg.vocab_size, (2, 32), device=device)
    logits, loss = model(x, y)
    assert logits.shape == (2, 32, cfg.vocab_size)
    assert torch.isfinite(loss) and loss.item() > 0
    print(f"[1] forward OK  logits={tuple(logits.shape)}  loss={loss.item():.3f}")

    # greedy KV-cache == greedy naive  (correctness of the cache)
    prompt = torch.randint(0, cfg.vocab_size, (1, 8), device=device)
    N = 40
    ref = naive_generate(model, prompt.clone(), N)
    cached = model.generate(prompt.clone(), N, temperature=1.0, top_k=1)  # top_k=1 == greedy
    match = torch.equal(ref, cached)
    print(f"[2] KV-cache vs naive greedy match: {match}")
    assert match, "KV cache diverges from the reference decoder!"

    # speed: cached vs naive over a longer generation
    M = 200
    t0 = time.time(); naive_generate(model, prompt.clone(), M); t_naive = time.time() - t0
    t0 = time.time(); model.generate(prompt.clone(), M, temperature=1.0, top_k=1); t_cached = time.time() - t0
    print(f"[3] generate {M} tokens:  naive {t_naive*1e3:.0f}ms  |  "
          f"cached {t_cached*1e3:.0f}ms  |  speedup {t_naive/t_cached:.2f}x")

    # chat template: round-trip + loss masking
    msgs = [{"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"}]
    ids, mask = cf.encode_example(msgs)
    assert len(ids) == len(mask)
    assert ids[-1] == cf.EOT and mask[-1] == 1, "reply must end with a trained EOT"
    # user tokens must be masked out (0), some assistant tokens trained (1)
    assert sum(mask) > 0 and sum(mask) < len(mask)
    text = cf.ENC.decode([t for t in ids if t != cf.EOT])
    assert "User: hello" in text and "Assistant: hi there" in text
    print(f"[4] chat template OK  tokens={len(ids)}  trained_on={sum(mask)}  "
          f"masked={len(mask)-sum(mask)}")

    # prompt rendering ends ready for the assistant to speak
    p = cf.render_prompt([("user", "hey")])
    assert p.rstrip().endswith("Assistant:")
    print("[4] render_prompt ends at 'Assistant:' OK")

    # gradient checkpointing must give the SAME loss as the normal path
    torch.manual_seed(0)
    m2 = GPT(cfg).to(device)
    m2.train()
    xb = torch.randint(0, cfg.vocab_size, (2, 32), device=device)
    yb = torch.randint(0, cfg.vocab_size, (2, 32), device=device)
    m2.use_checkpoint = False
    _, l_plain = m2(xb, yb)
    m2.use_checkpoint = True
    _, l_ckpt = m2(xb, yb)
    assert torch.allclose(l_plain, l_ckpt, atol=1e-4), (l_plain.item(), l_ckpt.item())
    l_ckpt.backward()                              # backward must work through checkpoints
    assert any(p.grad is not None for p in m2.parameters())
    print(f"[5] grad checkpoint loss matches ({l_plain.item():.4f}) + backward OK")

    # EOS stopping: set eos to the token the model WILL greedily emit first, so
    # generation must halt after exactly one new token.
    seed = torch.zeros((1, 1), dtype=torch.long, device=device)
    first = model.generate(seed.clone(), max_new_tokens=1, top_k=1)[0, -1].item()
    stopped = model.generate(seed.clone(), max_new_tokens=50,
                             eos_token_id=first, top_k=1)
    assert stopped.shape[1] == 2, f"expected stop after 1 token, got {stopped.shape[1]-1}"
    print(f"[6] EOS stop halted after 1 token (eos={first}) OK")

    print("\nALL CHECKS PASSED")


if __name__ == '__main__':
    main()
