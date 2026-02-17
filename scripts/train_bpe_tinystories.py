import sys
import os
import pickle
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cs336_basics.bpe_training import train_bpe

from config import TS_TRAIN_TEXT_PATH, TS_VOCAB_PATH, TS_MERGES_PATH, TS_VOCAB_SIZE, SPECIAL_TOKENS

if __name__ == "__main__":
    start_time = time.time()
    vocab, merges = train_bpe(
        input_path=TS_TRAIN_TEXT_PATH,
        vocab_size=TS_VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS)
    pickle.dump(vocab, open(TS_VOCAB_PATH, "wb"))
    pickle.dump(merges, open(TS_MERGES_PATH, "wb"))
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")

    # Print the longest string in vocab.keys()
    longest_key = max(vocab.keys(), key=lambda k: len(vocab[k]))
    print("Longest vocab value (bytes):", vocab[longest_key])
    print("Length:", len(vocab[longest_key]))
