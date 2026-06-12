#prepare_data.py — download a general web-text dataset, tokenize it with the
#GPT-2 BPE (tiktoken), and write train.bin / val.bin that the trainer memory-maps.
#
#Run ONCE on the Spark before training:
#    python prepare_data.py
#
#This mirrors nanoGPT's data prep. The 10BT sample is ~10B tokens (tens of GB on
#disk once tokenized) — make sure you have the space. Tokenization is CPU-bound
#and parallelized across NUM_PROC workers (the Spark has 20 CPU cores).
import os
import numpy as np
import tiktoken
from datasets import load_dataset      #pip install datasets
from tqdm import tqdm

# ----------------------- config -----------------------
DATASET        = "HuggingFaceFW/fineweb-edu"  #high-quality, filtered web text
DATASET_CONFIG = "sample-10BT"                #~10B tokens; "sample-100BT" for more
NUM_PROC       = 8                            #tokenization workers
VAL_FRACTION   = 0.0005                       #tiny held-out validation split
SEED           = 2357

enc = tiktoken.get_encoding("gpt2")


def process(example):
    ids = enc.encode_ordinary(example["text"])   #BPE token ids (ignores special tokens)
    ids.append(enc.eot_token)                     #<|endoftext|> marks doc boundaries
    return {"ids": ids, "len": len(ids)}


if __name__ == "__main__":
    #load the dataset (downloads + caches under ~/.cache/huggingface)
    ds = load_dataset(DATASET, name=DATASET_CONFIG, split="train", num_proc=NUM_PROC)

    #carve off a small validation split
    split = ds.train_test_split(test_size=VAL_FRACTION, seed=SEED, shuffle=True)
    split["val"] = split.pop("test")              #rename test -> val

    #tokenize every document in parallel
    tokenized = split.map(
        process,
        remove_columns=["text"],
        desc="tokenizing",
        num_proc=NUM_PROC,
    )

    #write each split to a flat uint16 binary (GPT-2 ids max at 50256, fits uint16)
    for name, dset in tokenized.items():
        arr_len = int(np.sum(dset["len"], dtype=np.uint64))
        filename = f"{name}.bin"
        arr = np.memmap(filename, dtype=np.uint16, mode="w+", shape=(arr_len,))
        total_batches = 1024
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
            batch = dset.shard(num_shards=total_batches, index=batch_idx,
                               contiguous=True).with_format("numpy")
            arr_batch = np.concatenate(batch["ids"])
            arr[idx: idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
        print(f"wrote {filename}: {arr_len:,} tokens")

    print("done. now train:  python DecoderOnlyLinuxGPT.py")
