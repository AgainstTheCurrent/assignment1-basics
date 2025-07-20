import sys
import os
import pickle
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cs336_basics.bpe_training import train_bpe

input_file = "../data/owt_train.txt"
output_vocab_file = "../data/owt-vocab.json"
output_merges_file = "../data/owt-merges.json"

if __name__ == "__main__":
    start_time = time.time()
    vocab, merges = train_bpe(
        input_path=input_file,
        vocab_size=32000,
        special_tokens=["<|endoftext|>"])
    pickle.dump(vocab, open(output_vocab_file, "wb"))
    pickle.dump(merges, open(output_merges_file, "wb"))
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")

    # Print the longest string in vocab.keys()
    longest_key = max(vocab.keys(), key=lambda k: len(vocab[k]))
    print("Longest vocab value (bytes):", vocab[longest_key])
    print("Length:", len(vocab[longest_key]))
