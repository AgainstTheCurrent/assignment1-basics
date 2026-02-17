import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle
from time import time
from cs336_basics.bpe_tokenizer import BPETokenizer
import numpy as np
import gc

ts_input_files = ("../data/TinyStoriesV2-GPT4-valid.txt", "../data/TinyStoriesV2-GPT4-train.txt")
owt_input_files = ("../data/owt_valid.txt", "../data/owt_train.txt")
ts_output_files = ("../data/TinyStoriesV2-GPT4-valid-tokens.bin",  "../data/TinyStoriesV2-GPT4-train-tokens.bin")
owt_output_files = ("../data/owt_valid-tokens.bin", "../data/owt_train-tokens.bin")

ts_vocab_file = "../data/TinyStoriesV2-GPT4-vocab.json"
ts_merges_file = "../data/TinyStoriesV2-GPT4-merges.json"
owt_vocab_file = "../data/owt-vocab.json"
owt_merges_file = "../data/owt-merges.json"


if __name__ == "__main__":
    special_tokens = ["<|endoftext|>"]

    tokenizer = BPETokenizer.from_files(ts_vocab_file, ts_merges_file, special_tokens=special_tokens)
    for ts_input, ts_output in zip(ts_input_files, ts_output_files):
        start_time = time()
        tokenizer.encode_file(ts_input, ts_output)
        end_time = time()
        print(f"Tokenization of {ts_input} completed in {end_time - start_time:.2f} seconds.")

    del tokenizer
    gc.collect()
    tokenizer = BPETokenizer.from_files(owt_vocab_file, owt_merges_file, special_tokens=special_tokens)
    for owt_input, owt_output in zip(owt_input_files, owt_output_files):
        start_time = time()
        tokenizer.encode_file(owt_input, owt_output)
        end_time = time()
        print(f"Tokenization of {owt_input} completed in {end_time - start_time:.2f} seconds.")
