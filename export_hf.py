"""export_hf.py — export a checkpoint to HuggingFace GPT-2 format for Ollama.

The model IS a GPT-2, so llama.cpp can convert it to GGUF and Ollama can run it.
This writes a folder (default ./hf_export/) with everything the converter needs:
config.json, pytorch_model.bin (weights remapped to HF names), and the GPT-2 BPE
tokenizer files (vocab.json + merges.txt, reconstructed from tiktoken — no
download required).

    python export_hf.py --ckpt gpt_chat.pt --out hf_export

Then (one-time, needs llama.cpp):
    git clone https://github.com/ggml-org/llama.cpp
    pip install -r llama.cpp/requirements.txt
    python llama.cpp/convert_hf_to_gguf.py ./hf_export --outfile model.gguf --outtype f16
    ollama create mychat -f hf_export/Modelfile
    ollama run mychat

HF GPT-2 stores the attention/MLP projection weights in Conv1D layout ([in, out],
the transpose of nn.Linear), so we transpose those four weights on the way out;
llama.cpp transposes them back. Attention has no bias in our model, so we emit
zeros (HF declares those biases).
"""
import os
import json
import argparse

import torch
import tiktoken

from model import GPT


# ---- GPT-2 byte<->unicode table (so raw bytes become printable vocab strings) ----
def bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def _bpe_merge(ranks, token, max_rank):
    """Re-derive the single merge that produced `token` (the lowest-rank pair)."""
    parts = [bytes([b]) for b in token]
    while True:
        best_i, best_rank = None, None
        for i in range(len(parts) - 1):
            r = ranks.get(parts[i] + parts[i + 1])
            if r is not None and (best_rank is None or r < best_rank):
                best_i, best_rank = i, r
        if best_rank is None or best_rank >= max_rank:
            break
        parts = parts[:best_i] + [parts[best_i] + parts[best_i + 1]] + parts[best_i + 2:]
    return parts


def write_tokenizer(out_dir):
    """Reconstruct GPT-2 vocab.json + merges.txt from tiktoken's BPE ranks."""
    enc = tiktoken.get_encoding('gpt2')
    ranks = enc._mergeable_ranks
    byte_enc = bytes_to_unicode()

    def tok_str(b):
        return "".join(byte_enc[x] for x in b)

    # vocab.json: token-string -> id  (+ the special end-of-text token)
    vocab = {tok_str(b): i for b, i in ranks.items()}
    vocab["<|endoftext|>"] = enc.eot_token

    # merges.txt: each multi-byte token implies one merge; order by rank
    merges = []
    for token, rank in ranks.items():
        if len(token) == 1:
            continue
        pair = _bpe_merge(ranks, token, max_rank=rank)
        assert len(pair) == 2, f"could not recover merge for {token!r}"
        merges.append((rank, tok_str(pair[0]), tok_str(pair[1])))
    merges.sort()

    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    with open(os.path.join(out_dir, "merges.txt"), "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")
        for _, a, b in merges:
            f.write(f"{a} {b}\n")
    with open(os.path.join(out_dir, "tokenizer_config.json"), "w") as f:
        json.dump({"model_type": "gpt2", "bos_token": "<|endoftext|>",
                   "eos_token": "<|endoftext|>", "unk_token": "<|endoftext|>"}, f)

    # sanity: vocab/merge counts and an id round-trip against tiktoken
    assert len(vocab) == 50257, f"vocab size {len(vocab)} != 50257"
    assert len(merges) == 50000, f"merge count {len(merges)} != 50000"
    id_to_str = {i: s for s, i in vocab.items()}
    sample = enc.encode_ordinary("Hello, world! The model exports cleanly.")
    assert "".join(id_to_str[i] for i in sample), "id->token mapping is broken"
    return len(vocab), len(merges)


def export(ckpt_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    model, _ = GPT.from_checkpoint(ckpt_path)
    sd = model.state_dict()
    cfg = model.cfg
    V = 50257                                  # real GPT-2 vocab (drop padded rows)

    hf = {}
    hf["transformer.wte.weight"] = sd["token_embedding_table.weight"][:V].clone()
    hf["transformer.wpe.weight"] = sd["position_embedding_table.weight"].clone()
    hf["transformer.ln_f.weight"] = sd["ln_f.weight"]
    hf["transformer.ln_f.bias"]   = sd["ln_f.bias"]
    hf["lm_head.weight"]          = sd["token_embedding_table.weight"][:V].clone()  # tied

    for i in range(cfg.n_layer):
        p = f"blocks.{i}."
        h = f"transformer.h.{i}."
        hf[h + "ln_1.weight"] = sd[p + "ln1.weight"]
        hf[h + "ln_1.bias"]   = sd[p + "ln1.bias"]
        hf[h + "ln_2.weight"] = sd[p + "ln2.weight"]
        hf[h + "ln_2.bias"]   = sd[p + "ln2.bias"]
        # Conv1D layout = nn.Linear weight transposed to [in, out]
        hf[h + "attn.c_attn.weight"] = sd[p + "sa.c_attn.weight"].t().contiguous()
        hf[h + "attn.c_proj.weight"] = sd[p + "sa.c_proj.weight"].t().contiguous()
        hf[h + "mlp.c_fc.weight"]    = sd[p + "ffwd.c_fc.weight"].t().contiguous()
        hf[h + "mlp.c_proj.weight"]  = sd[p + "ffwd.c_proj.weight"].t().contiguous()
        # attention has no bias in our model -> zeros; MLP biases are real
        hf[h + "attn.c_attn.bias"] = torch.zeros(3 * cfg.n_embd)
        hf[h + "attn.c_proj.bias"] = torch.zeros(cfg.n_embd)
        hf[h + "mlp.c_fc.bias"]    = sd[p + "ffwd.c_fc.bias"]
        hf[h + "mlp.c_proj.bias"]  = sd[p + "ffwd.c_proj.bias"]

    hf = {k: v.to(torch.float32).contiguous() for k, v in hf.items()}
    torch.save(hf, os.path.join(out_dir, "pytorch_model.bin"))

    config = {
        "architectures": ["GPT2LMHeadModel"],
        "model_type": "gpt2",
        "vocab_size": V,
        "n_positions": cfg.block_size,
        "n_ctx": cfg.block_size,
        "n_embd": cfg.n_embd,
        "n_layer": cfg.n_layer,
        "n_head": cfg.n_head,
        "layer_norm_epsilon": 1e-5,
        "activation_function": "gelu_new",
        "bos_token_id": 50256,
        "eos_token_id": 50256,
        "tie_word_embeddings": True,
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    nv, nm = write_tokenizer(out_dir)

    # A Modelfile matching the chat template (so Ollama formats turns + stops right).
    modelfile = (
        "FROM ./model.gguf\n\n"
        'TEMPLATE """The following is a conversation with a helpful, friendly AI assistant.\n'
        "{{ range .Messages }}{{ if eq .Role \"user\" }}User: {{ .Content }}\n"
        "{{ else if eq .Role \"assistant\" }}Assistant: {{ .Content }}\n"
        '{{ end }}{{ end }}Assistant: """\n\n'
        "PARAMETER temperature 0.8\n"
        "PARAMETER top_p 0.95\n"
        f"PARAMETER num_ctx {cfg.block_size}\n"
        'PARAMETER stop "<|endoftext|>"\n'
        'PARAMETER stop "\\nUser:"\n'
    )
    with open(os.path.join(out_dir, "Modelfile"), "w") as f:
        f.write(modelfile)

    print(f"exported -> {out_dir}/")
    print(f"  config.json, pytorch_model.bin ({len(hf)} tensors), "
          f"vocab.json ({nv}), merges.txt ({nm}), Modelfile")
    print("\nnext (needs llama.cpp):")
    print(f"  python llama.cpp/convert_hf_to_gguf.py ./{out_dir} "
          f"--outfile {out_dir}/model.gguf --outtype f16")
    print(f"  ollama create mychat -f {out_dir}/Modelfile && ollama run mychat")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="gpt_chat.pt")
    ap.add_argument("--out", default="hf_export")
    args = ap.parse_args()
    if not os.path.exists(args.ckpt):
        raise SystemExit(f"{args.ckpt} not found -- train/fine-tune first.")
    export(args.ckpt, args.out)
