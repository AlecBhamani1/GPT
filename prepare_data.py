#prepare_data.py — tokenize text with the GPT-2 BPE (tiktoken) and write
#train.bin / val.bin that the trainer memory-maps.
#
#Two modes:
#  1. Big web-text pretraining (default): streams FineWeb-Edu from HuggingFace.
#         python prepare_data.py
#  2. A LOCAL text file (great for quick/easy training on your own data):
#         python prepare_data.py --file mydata.txt
#     No download, no `datasets` dependency — just point it at a .txt.
#
#The 10BT sample is ~10B tokens (tens of GB once tokenized) — make sure you have
#the space. Tokenization is CPU-bound and parallelized across NUM_PROC workers.
import os
import sys
import argparse
import numpy as np
import tiktoken
from tqdm import tqdm

# ----------------------- config -----------------------
DATASET        = "HuggingFaceFW/fineweb-edu"  #high-quality, filtered web text
DATASET_CONFIG = "sample-10BT"                #~10B tokens; "sample-100BT" for more
NUM_PROC       = 8                            #tokenization workers
VAL_FRACTION   = 0.0005                       #tiny held-out validation split
SEED           = 2357

enc = tiktoken.get_encoding("gpt2")


def prepare_local_file(path, val_fraction=0.01):
    """Tokenize one local text file into train.bin / val.bin. Lightweight path
    for training on your own corpus without downloading a big dataset."""
    with open(path, encoding='utf-8', errors='ignore') as f:
        text = f.read()
    ids = np.array(enc.encode_ordinary(text) + [enc.eot_token], dtype=np.uint16)
    n_val = max(1, int(len(ids) * val_fraction))
    splits = {'train': ids[:-n_val], 'val': ids[-n_val:]}
    for name, arr in splits.items():
        arr.tofile(f"{name}.bin")
        print(f"wrote {name}.bin: {len(arr):,} tokens")
    print("done. now train:  python train.py   (try GPT_CONFIG=tiny first)")


def process(example):
    ids = enc.encode_ordinary(example["text"])   #BPE token ids (ignores special tokens)
    ids.append(enc.eot_token)                     #<|endoftext|> marks doc boundaries
    return {"ids": ids, "len": len(ids)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="tokenize this local .txt instead of FineWeb-Edu")
    args = ap.parse_args()

    if args.file:
        prepare_local_file(args.file)
        sys.exit(0)

    from datasets import load_dataset   #only needed for the big-download path

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

    print("done. now train:  python train.py")
