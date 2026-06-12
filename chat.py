#chat.py — load the trained weights and prompt the model interactively.
#
#Run this AFTER training has produced gpt_model.pt:
#    python prepare_data.py            # one-time: build train.bin / val.bin
#    python DecoderOnlyLinuxGPT.py     # train (checkpoints to gpt_model.pt as it goes)
#    python chat.py                    # load + prompt, as often as you like
#
#This is a text CONTINUATION model, not an instruction-following assistant: it
#continues your prompt in the style of general web text it was trained on.
import torch
import tiktoken
import DecoderOnlyLinuxGPT as gpt   #imports the model classes + config; does NOT train

CHECKPOINT  = 'gpt_model.pt'
MAX_NEW     = 300        #tokens to generate per prompt
TEMPERATURE = 0.8        #lower = more focused, higher = more random
TOP_K       = 200        #sample only from the top-k most likely tokens
device      = gpt.device

#load the checkpoint. weights_only=False because it stores a small config dict too;
#safe since we created the file.
ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
cfg = ckpt['config']

#point the model's module-level globals at the values this checkpoint was trained
#with, BEFORE building the model (keeps architecture in sync with the weights).
gpt.vocab_size = cfg['vocab_size']
gpt.n_embd     = cfg['n_embd']
gpt.n_head     = cfg['n_head']
gpt.n_layer    = cfg['n_layer']
gpt.block_size = cfg['block_size']
gpt.dropout    = cfg['dropout']

model = gpt.BigramLanguageModel()
model.load_state_dict(ckpt['model_state'])
model.to(device)
model.eval()

#GPT-2 BPE: byte-level, so ANY text encodes — no unknown-character handling needed.
enc = tiktoken.get_encoding(ckpt.get('tokenizer', 'gpt2'))
encode = lambda s: enc.encode_ordinary(s)
decode = lambda l: enc.decode(l)

print(f"Loaded {CHECKPOINT} (trained {ckpt.get('iter', '?')} steps). "
      "Type a prompt; the model continues it. 'quit' to exit.\n")

while True:
    try:
        prompt = input("you> ")
    except (EOFError, KeyboardInterrupt):
        print()
        break
    if prompt.strip() in ("quit", "exit"):
        break

    if prompt == "":
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    else:
        idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

    with torch.autocast(device_type=gpt.device_type, dtype=gpt.dtype):
        out = model.generate(idx, max_new_tokens=MAX_NEW,
                             temperature=TEMPERATURE, top_k=TOP_K)
    print(decode(out[0].tolist()) + "\n")
