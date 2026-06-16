# DecoderOnlyGPT — a from-scratch GPT you can train into a chatbot

Originally hand-written following Andrej Karpathy's "Let's build GPT from scratch"
([video](https://www.youtube.com/watch?v=kCc8FmEb1nY)), now a clean GPT-2 with the
pieces needed to **train a small chatbot that messages back and forth** — and export
it to run in **Ollama**.

It's a decoder-only transformer over GPT-2 BPE tokens (`tiktoken`). Pretraining
makes it a text *continuation* model; chat fine-tuning (`finetune_chat.py`) teaches
it to hold a conversation.

## Quickstart on a new machine (from scratch)

Works on Linux, macOS, or Windows. A CUDA GPU is needed to train the full `gpt2`
size in reasonable time; a laptop (CPU/Apple-MPS) is fine for the `tiny` preset.

### 0. Prerequisites
- **Git** and **Python 3.10+** (`python3 --version`).
- For full-size training: an NVIDIA **CUDA GPU** (24GB+ comfortably fits 124M).
- Disk: a few hundred MB for the local-file path; **tens of GB** for FineWeb-Edu.

### 1. Clone and create a virtual environment
```bash
git clone https://github.com/AlecBhamani1/GPT.git
cd GPT

python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
> **GPU users — read this.** The `torch` from `requirements.txt` is the default
> build (CPU, plus Apple-MPS on Macs). To train on an NVIDIA GPU, install the build
> that matches your CUDA version from the selector at
> <https://pytorch.org/get-started/locally/>, e.g.:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu124
> ```
> Verify the GPU is seen: `python -c "import torch; print(torch.cuda.is_available())"`
> should print `True`.

Optional sanity check (builds a tiny model, runs a few forward/backward steps and a
short generation — finishes in seconds):
```bash
python smoke_test.py
```

### 3. Get training data → `train.bin` / `val.bin`
Pick **one**:

```bash
# A) Quick: tokenize your own plain-text file (no download, no `datasets` needed)
python prepare_data.py --file mydata.txt

# B) Full: stream the FineWeb-Edu web-text corpus (~10B tokens, tens of GB on disk)
python prepare_data.py
```
(No corpus handy? `python prepare_data.py --file linux_input.txt` uses the bundled
sample text — enough to exercise the whole pipeline.)

### 4. Pretrain the base model → `gpt_model.pt`
Choose a size with `GPT_CONFIG` (`tiny | small | gpt2`, default `gpt2`):

```bash
# Laptop / quick smoke run (minutes):
GPT_CONFIG=tiny python train.py

# Full GPT-2-small on a GPU — run it detached and watch the log:
nohup python -u train.py > train.log 2>&1 &
tail -f train.log
```
Training is **resumable**: it checkpoints to `gpt_model.pt` every few hundred steps,
so rerunning the same command picks up where it left off. Out of memory? add
`GRAD_CKPT=1` (gradient checkpointing) and/or lower `BATCH_SIZE`.

### 5. Fine-tune into a chat assistant → `gpt_chat.pt`
Provide conversations as a JSONL file — one JSON object per line:
```json
{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello! How can I help?"}]}
```
Then:
```bash
python finetune_chat.py --data sample_chat.jsonl     # swap in your own .jsonl
```
(`sample_chat.jsonl` is a tiny demo — replace it with real conversations for a
useful bot. The loader validates your file and reports the exact line on any error.)

### 6. Chat
```bash
python chat.py                       # back-and-forth, remembers the conversation
```

### 7. (Optional) Measure quality / export to Ollama
```bash
python eval.py --ckpt gpt_chat.pt --chat sample_chat.jsonl   # perplexity (lower=better)
python export_hf.py --ckpt gpt_chat.pt --out hf_export       # then see "Running in Ollama"
```

## What's here

| File | Role |
|------|------|
| `model.py` | GPT architecture + `GPTConfig` + presets, **KV-cache** generation (top-k/top-p/repetition-penalty/EOS), checkpoint helpers. Imports with no side effects. |
| `train.py` | Pretraining loop (bf16/TF32 on CUDA, MPS & CPU supported, cosine LR, grad accumulation, resumable). |
| `chat_format.py` | The conversation template shared by fine-tuning and chat (loss-masked so the model learns to *reply*, not echo). |
| `finetune_chat.py` | Supervised fine-tuning on JSONL conversations → `gpt_chat.pt`. Holds out a val split, saves the **best** checkpoint by val loss, early-stops. |
| `chat.py` | Interactive chat REPL with history, streaming, and end-of-turn stopping. |
| `prepare_data.py` | Tokenize a local `.txt` (`--file`) or stream FineWeb-Edu → `train.bin` / `val.bin`. |
| `export_hf.py` | Export a checkpoint to HuggingFace GPT-2 format + a `Modelfile` for **Ollama**. |
| `eval.py` | Measure perplexity (lower = better) on a `.bin` or assistant replies in a chat JSONL. |
| `smoke_test.py` | Fast correctness/speed checks (KV cache vs reference, template, masking, checkpointing). |

## Config presets (`GPT_CONFIG`)

| Preset | dims | params | for |
|--------|------|--------|-----|
| `tiny`  | 128d / 4L / 4H, ctx 256  | ~7M   | laptop CPU/MPS, smoke tests, demos |
| `small` | 512d / 8L / 8H, ctx 512  | ~30M  | a single consumer GPU overnight |
| `gpt2`  | 768d / 12L / 12H, ctx 1024 | ~124M | the original DGX Spark target |

Tweak hyperparameters per run via env vars: `BATCH_SIZE`, `GRAD_ACCUM`, `MAX_ITERS`,
`COMPILE`, `CKPT`, and `GRAD_CKPT=1` (gradient checkpointing — much less memory, so
a bigger model/context fits on a small GPU, ~20-30% slower).

## Chatting

`chat.py` auto-loads `gpt_chat.pt` (chat mode) if present, else `gpt_model.pt`
(plain continuation). In chat mode it renders each turn with the template, streams
the reply token-by-token, and stops at the end-of-turn token. `/reset` clears
history (and slides the window so long conversations keep working), `/quit` exits.
Sampling is tunable: `--temperature`, `--top-k`, `--top-p`, `--repetition-penalty`.
For scripting, `python chat.py --prompt "hi"` answers one message and exits. Give
the bot a persona with `--system "You are a terse pirate."` (works best if it was
fine-tuned with a matching style). Sampling args are validated, so bad values fail
fast instead of producing junk.

Track quality with `python eval.py --ckpt gpt_chat.pt --chat sample_chat.jsonl`
(assistant-reply perplexity) or `--bin val.bin` (raw LM perplexity).

## Running in Ollama

The model *is* a GPT-2, so `llama.cpp` can convert it to GGUF and Ollama can run it:

```bash
python export_hf.py --ckpt gpt_chat.pt --out hf_export      # self-contained HF dir

git clone https://github.com/ggml-org/llama.cpp
pip install -r llama.cpp/requirements.txt
python llama.cpp/convert_hf_to_gguf.py ./hf_export --outfile hf_export/model.gguf --outtype f16

ollama create mychat -f hf_export/Modelfile
ollama run mychat
```

`export_hf.py` writes `config.json`, the weights (remapped to HF tensor names, with
the Conv1D-layout transposes llama.cpp expects), the GPT-2 tokenizer (`vocab.json` +
`merges.txt`, reconstructed from tiktoken — no download), and a `Modelfile` whose
chat template + stop tokens match how the model was fine-tuned.

## Performance notes

- **KV cache**: generation reuses cached keys/values instead of recomputing the full
  context each token — bit-exact vs the naive decoder, and faster (the gap widens
  with context length and on CUDA with flash attention).
- **Precision**: bf16 autocast on CUDA (wide range, no loss scaling); fp32 on
  MPS/CPU (correct and dodges flaky half-precision).
- **Throughput** on the DGX Spark is memory-bandwidth bound at 124M; raise
  `BATCH_SIZE` and lower `GRAD_ACCUM` to use the 128GB.

## Deactivating

When you're done, `deactivate` exits the virtualenv.

> Legacy: `linux_input.txt` was the original character-level corpus — handy as a
> local `--file` for a quick `tiny` run, but not used by the default web-text path.
