# DecoderOnlyGPT — a from-scratch GPT, scaled to GPT-2 size

Originally hand-written following Andrej Karpathy's "Let's build GPT from scratch"
([video](https://www.youtube.com/watch?v=kCc8FmEb1nY)), now scaled up for
**general web-text pretraining at GPT-2-small size (~124M params)** on a single
**NVIDIA DGX Spark**.

It's a decoder-only transformer trained to predict the next **BPE token** (GPT-2
tokenizer via `tiktoken`). That makes it a text *continuation* model — you prompt it
and it continues in the style of its training data. It is **not** an instruction-tuned
chat assistant.

## Pipeline

Three steps, in order:

```bash
pip install -r requirements.txt

python prepare_data.py          # 1. download FineWeb-Edu, tokenize -> train.bin / val.bin (one-time)
python DecoderOnlyLinuxGPT.py   # 2. train (checkpoints to gpt_model.pt; resumable)
python chat.py                  # 3. prompt the trained model
```

### 1. Data — `prepare_data.py`
Downloads the **FineWeb-Edu** `sample-10BT` split (~10B tokens of high-quality web
text), tokenizes it with the GPT-2 BPE, and writes flat `uint16` token files
(`train.bin`, `val.bin`) that training memory-maps. Expect tens of GB on disk. Swap
`DATASET_CONFIG` to `sample-100BT` for more data.

### 2. Training — `DecoderOnlyLinuxGPT.py`
GPT-2-small config (768-dim, 12 layers, 12 heads, 1024-token context). Uses bf16
autocast, `torch.compile`, TF32, a cosine LR schedule with warmup, gradient
clipping, weight decay, and gradient accumulation (~0.5M tokens / effective batch).

**Long runs are resumable.** Checkpoints (model + optimizer state) are written to
`gpt_model.pt` every `eval_interval` steps. If training is interrupted, just rerun
the same command — it picks up from the last checkpoint. To run detached and watch
the log live:

```bash
nohup python -u DecoderOnlyLinuxGPT.py > train.log 2>&1 &
tail -f train.log
```

Loss starts near ~10.8 (random guessing over a 50k vocab) and should fall to ~3–4.

### 3. Chat — `chat.py`
Loads `gpt_model.pt` and gives an interactive prompt with temperature / top-k
sampling. Importing the trainer here does **not** trigger training (it's guarded by
`if __name__ == '__main__'`), so startup is fast.

## DGX Spark notes

- **Memory:** 128GB unified — you won't OOM at 124M. Raise `batch_size` and lower
  `grad_accum_steps` to use it.
- **Precision:** **bf16** is the right call for training on Blackwell (wide dynamic
  range, no loss scaling, fully accelerated). fp16 is strictly worse; fp8 (via NVIDIA
  Transformer Engine) is the only faster path but adds complexity — not needed at
  124M. fp4 is an inference format, not for training.
- **Throughput:** the hand-built attention (one module per head) is the main
  bottleneck. See the note at the top of the source about batching the heads /
  `F.scaled_dot_product_attention` (flash attention) for a large speedup.

## Setup (virtual environment)

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt
deactivate                       # when done
```

### GPU note
PyTorch from `requirements.txt` runs on CPU everywhere and on CUDA GPUs on
Linux/Windows. For a build matched to your CUDA version, use the selector at
https://pytorch.org/get-started/locally/. Training auto-detects CUDA via
`torch.cuda.is_available()`.

## Files
- `prepare_data.py` — build the tokenized dataset (`train.bin` / `val.bin`)
- `DecoderOnlyLinuxGPT.py` — model definition + training loop
- `chat.py` — interactive generation from a checkpoint
- `requirements.txt` — dependencies

> Legacy: `linux_input.txt` was the original character-level corpus and is no longer
> used by the BPE pipeline (kept for reference). The training script keeps its
> original name so `chat.py`'s import still works.
