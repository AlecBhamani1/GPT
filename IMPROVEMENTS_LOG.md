# Improvements Log

Goal: turn the GPT-2-small *continuation* model into a real **chatbot** that can be
trained and chat back and forth — faster, easier to train, runnable via Ollama.

Each iteration records what changed, why, and measured metrics. Validation runs on
this machine (Apple Silicon, MPS available, no CUDA) with a `tiny` config; the
`gpt2` config is unchanged behaviorally and still targets the DGX Spark.

---

## Baseline (commit 67b6851)

- `DecoderOnlyLinuxGPT.py` — GPT-2-small (124M) **continuation** pretraining only.
- `chat.py` — single-shot prompt continuation, no conversation history, no stop token.
- Generation: `generate()` re-runs the **full context every token** → O(T³) total, no KV cache.
- Device detection: `cuda` or `cpu` only — **MPS (Apple GPU) ignored**.
- No chat template, no fine-tuning path, no Ollama/GGUF export.
- Misleading class name `BigramLanguageModel` (it is a full GPT).

Gap: the stated goal is a *chatbot*, but the code only does next-token continuation.

---

## Iteration 1 — make it a trainable, KV-cached chatbot (+ Ollama export)

### Architecture / restructure
- Split the monolithic `DecoderOnlyLinuxGPT.py` into a clean, importable
  `model.py` (`GPTConfig` dataclass + `GPT` + presets) and `train.py` (pretraining).
  Renamed the misleading `BigramLanguageModel` → `GPT` (kept an alias). Checkpoints
  now carry their full config, so chat/finetune rebuild the exact architecture —
  the fragile "set globals before constructing" dance is gone.
- **KV cache** in `generate()`: incremental decoding reuses cached keys/values
  instead of recomputing the whole context every token. Added top-p (nucleus),
  repetition penalty, EOS stopping, and a streaming callback.
- **MPS support + adaptive precision**: device picks CUDA→MPS→CPU (Apple GPU was
  previously ignored); bf16 autocast only on CUDA, fp32 elsewhere.

### Chatbot capability (the core gap)
- `chat_format.py`: shared, loss-masked conversation template (User/Assistant turns,
  `<|endoftext|>` as end-of-turn). Loss is trained only on assistant replies.
- `finetune_chat.py`: supervised fine-tuning on JSONL conversations → `gpt_chat.pt`.
- `chat.py` rewritten: real back-and-forth — keeps history, renders the template,
  streams the reply, stops at end-of-turn, `/reset` + `/quit`.

### Easier to train
- Config presets `tiny | small | gpt2` so you don't need a DGX Spark to train
  *something*; hyperparameters scale with the preset.
- `prepare_data.py --file mydata.txt`: tokenize a local text file with no 10GB
  download and no `datasets` dependency.

### Ollama
- `export_hf.py`: export to HF GPT-2 format (correct Conv1D transposes, zero attn
  biases, tied lm_head) + reconstruct the GPT-2 tokenizer from tiktoken + emit a
  `Modelfile` whose template/stops match fine-tuning. → `convert_hf_to_gguf.py` →
  `ollama create`.

### Validation & metrics (this machine: Apple Silicon, MPS, fp32)
| Check | Result |
|-------|--------|
| `smoke_test.py` all checks | **PASS** |
| KV-cache greedy vs naive full-recompute | **bit-exact match** |
| Gen speedup (200 tok, 7M model, ctx 128) | **~2–2.9×** faster |
| Gen speedup (30M model, ctx 512) | **1.6–2.1×** faster |
| Pretrain `tiny`, 200 steps on local text | loss **10.80 → 4.29** |
| Fine-tune chat, 400 steps | loss **9.15 → 5.95**, runs in chat mode |
| Full pipeline prepare→train→finetune→chat | **works end-to-end** |
| HF export | 53 tensors, config.json valid |
| Tokenizer reconstruction vs tiktoken | **bit-exact** on all test strings |

Output text is gibberish at these toy sizes/step counts (7M params, seconds of
training) — the point of the smoke runs is to validate the *machinery*. Real
quality needs the `gpt2` preset on a GPU with the FineWeb-Edu corpus, then chat
fine-tuning on a real conversation dataset.

### Notes / future
- KV cache uses `torch.cat` to grow the cache (simple, standard). A preallocated
  ring buffer would cut per-step allocation — matters most on MPS; revisit if
  generation latency becomes a bottleneck.
- The GGUF conversion itself needs `llama.cpp` (external) and wasn't run here; the
  export is verified structurally (tensor names/shapes/config) and the tokenizer is
  verified bit-exact, which are the parts this repo controls.

---

## Iteration 2 — training efficiency, long-chat robustness, scriptability

Design constraint locked in: the model stays **GPT-2-architecture** so the Ollama
GGUF export keeps working. So no RMSNorm/rotary swaps — gains come from training
efficiency, chat UX, and tooling instead.

- **Gradient checkpointing** (`GRAD_CKPT=1 python train.py`): recompute block
  activations in the backward pass instead of storing them (~√n_layer less
  activation memory) so a bigger model / longer context fits on a small GPU.
  `GPT.use_checkpoint` flag; only active during training without the KV cache.
- **Sliding-window chat context**: `chat.py` now drops the oldest turns until the
  prompt + reply headroom fit `block_size`, so long conversations keep working
  instead of silently hitting the context cap.
- **One-shot mode**: `python chat.py --prompt "hi"` answers once and exits —
  scriptable, testable, pipeable. Guarded so empty/`/reset` can't infinite-loop.
- **Bigger sample dataset**: 8 → 16 demo conversations for a more useful fine-tune
  starting point (real use still = bring your own JSONL).
- Removed a dead `stop_ids`/unused import in `chat.py`.

### Validation & metrics (Apple Silicon, MPS, fp32)
| Check | Result |
|-------|--------|
| `smoke_test.py` (now 6 checks) | **ALL PASS** |
| Grad-checkpoint loss == normal-path loss | **match @ atol 1e-4**, backward OK |
| EOS stop (eos = model's own greedy token) | **halts after exactly 1 token** |
| KV-cache greedy vs naive (re-verified) | **bit-exact** |
| Pretrain `tiny` with `GRAD_CKPT=1` | trains, loss starts ~10.8 (parity) |
| Fine-tune → `chat.py --prompt` one-shot | **generates + exits cleanly (rc 0)** |

### Still open / not worth doing
- Preallocated KV-cache ring buffer: prototyped mentally; on MPS the current
  `torch.cat` path already wins and the gain is marginal at these sizes — deferred
  until it's a measured bottleneck (CUDA + long context is where it'd pay off).
- Architecture quality upgrades (RMSNorm, RoPE, GQA) are intentionally **excluded**
  — they would break GPT-2 GGUF/Ollama compatibility, which is an explicit goal.

---

## Iteration 3 — better fine-tuning (validation, best-checkpoint, early stop)

**Constraint (user):** do NOT train/run the model on this machine ("it will not
work"). So from here, all validation is **static only** — syntax, imports, and
pure-logic unit checks; no `train.py`/`finetune_chat.py`/`smoke_test.py` runs.

- `finetune_chat.py` reworked:
  - Loads conversations per-example, **holds out a val split** (`--val-fraction`,
    default 0.15) of whole conversations when there are ≥4; otherwise validates on
    the train stream so a best-checkpoint is still chosen sensibly.
  - **Saves the best checkpoint by val loss**, not the last step (`val_loss` stored
    in the checkpoint), with **early stopping** (`--patience`, default 5 evals).
  - Logs train + val loss each eval.
- `chat.py` now shows the checkpoint's val loss on load (when present).
- README updated; `flatten()` factored out for clean per-stream padding.

### Validation (static only — no model executed, per constraint)
| Check | Result |
|-------|--------|
| All edited files `ast.parse` | **OK** |
| `import chat_format`, `import model` | **OK** |
| `load_dataset` on sample_chat.jsonl | 16 conversations parsed |
| `flatten()` ids/mask aligned, padded ≥block, mask∈{0,1} | **OK** (690 tok, 297 trained) |
| Train/val split disjoint + covers all + val≥1 | **OK** (14 train / 2 val) |

No model forward/backward/generation was run on this machine. The fine-tuning
loop's correctness rests on static review + the pure-logic checks above; it should
be exercised on the CUDA training box.

---

## Iteration 4 — quality metrics + chat polish (static-validated)

- **`eval.py`** (new): perplexity tool to actually answer "is it better?" — raw LM
  perplexity over a `.bin`, or **assistant-token-only** perplexity over a chat JSONL
  (same loss mask as fine-tuning). Deterministic contiguous coverage so runs are
  comparable; token-weighted aggregation across batches.
- **`chat.py --system "..."`**: override the persona/system preamble at chat time
  (threads through the existing `chat_format` template).
- **Sampling-arg validation** in `chat.py`: temperature/top-p/top-k/max-new/
  repetition-penalty are range-checked — bad values fail fast instead of emitting
  junk or crashing deep in generation.
- **Friendly `GPT_CONFIG` error** in `train.py` (was a raw `KeyError`).
- README documents `eval.py`, `--system`, and perplexity tracking.

### Validation (static only — no model executed, per constraint)
| Check | Result |
|-------|--------|
| All 9 modules `ast.parse` | **OK** |
| `eval.iter_blocks` shapes/offset/coverage | **OK** (9 blocks → batches 4,4,1; y=x+1) |
| Token-weighted perplexity math | **OK** (== exp(1.25)) |
| `eval.py` `--bin`/`--chat` mutual-exclusion guard | **OK** |
| Custom `--system` changes prompt tokens + still ends at `Assistant:` | **OK** |
| Invalid `GPT_CONFIG` → friendly message; valid imports clean | **OK** |

### Stopping assessment
The original goals are all met: trains (presets + local data + grad checkpointing),
chats back and forth (template + SFT + conversational REPL with history/streaming/
stop), faster generation (KV cache), Ollama path (verified HF export + tokenizer),
and now measurable quality (`eval.py`). Remaining ideas are increasingly marginal
(e.g. preallocated KV ring buffer — deferred as not worth it at these sizes). Absent
the ability to train/run here to find real regressions, further edits risk churn
over value. **Treating the core task as complete; will keep the loop light.**

---

## Iteration 5 — friendly fine-tuning data validation (final polish)

- `finetune_chat.load_dataset` now validates every JSONL line and fails with a
  clear, **line-numbered** message instead of crashing deep in training: bad JSON,
  missing `messages` key, empty/non-list messages, missing `role`/`content`,
  invalid role, blank content, and "no assistant reply to train on" are all caught
  up front. New `_validate_messages` helper.

### Validation (static only — no model executed, per constraint)
| Check | Result |
|-------|--------|
| `finetune_chat.py` parses | **OK** |
| Valid `sample_chat.jsonl`-style line accepted | **OK** |
| 8 malformed cases each raise the right friendly error | **OK** |
| Error reports the correct line number (line 3 of 3) | **OK** |

---

## Fix — resume across the refactor (optimizer-state layout)

Reported on the CUDA box: resuming a pre-refactor `gpt_model.pt` (step 3000) crashed
in `optimizer.step()` with `tensor a (3072) must match tensor b (768)`. Cause: the
old `Block` registered submodules in a different order, so the optimizer state
(keyed by parameter *position*) paired Adam buffers with the wrong params. Model
*weights* were fine (matched by name). Fix in `train.py`: after loading optimizer
state, verify each `exp_avg` shape matches its parameter; on any mismatch, keep the
loaded weights and `start_iter` but reset the optimizer (re-warms quickly). Validated
with a toy 2-param optimizer (detector flags the 3072-vs-768 corruption, passes a
correct state). No model trained/run on the dev machine.

---

## Loop concluded (after iteration 5)

The stopping criterion is effectively met. All stated goals are delivered and every
change is statically validated:

- **Trains / easier to train:** `tiny|small|gpt2` presets, local-`.txt` data path,
  gradient checkpointing, resumable pretraining, SFT with val split +
  best-checkpoint + early stop, and now friendly data validation.
- **Chats back and forth:** chat template with loss masking, conversational REPL
  with history, sliding-window context, streaming, end-of-turn stop, persona
  (`--system`), and one-shot mode.
- **Faster:** KV-cache generation (bit-exact vs naive, measured faster).
- **Ollama:** verified HF GPT-2 export (correct transposes, tied head) + bit-exact
  tokenizer reconstruction + matching Modelfile.
- **Measurable:** `eval.py` perplexity (raw + assistant-only).

Remaining ideas (preallocated KV ring buffer, fp8/quantized training, a larger
curated chat dataset, real end-to-end GGUF+Ollama run) all require either training
hardware or running the model — out of scope on this machine. Further blind edits
would be churn, so the autonomous loop is being wound down. Re-run `/loop` with a
specific new goal to resume.

---
