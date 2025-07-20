from collections import defaultdict
from queue import PriorityQueue
from typing import BinaryIO
import os
import regex as re
from cs336_basics.pretokenization import get_word_freq_mp

NUM_PROCESSES = os.cpu_count() or 2
SPLIT_SPECIAL_TOKEN = "<|endoftext|>".encode("utf-8")

class ReverseOrder:
    def __init__(self, val) -> None:
        self.val = val

    def __lt__(self, other) -> bool:
        return self.val > other.val

    def __repr__(self) -> str:
        return repr(self.val)

def init_vocab(vocab_size: int, special_token_bytes: list[bytes]) -> dict[int, bytes]:
    special_len = len(special_token_bytes)
    total_len = special_len + 256
    if vocab_size < total_len:
        raise ValueError("vocab_size must be at least 256 + number of special tokens")

    # Initialize the vocabulary with single-byte tokens
    vocab = {i + special_len: bytes([i]) for i in range(256)}
    idx = 0
    for token_bytes in special_token_bytes:
        vocab[idx] = token_bytes
        idx += 1
    return vocab

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Given a path to an input text file, trains a (byte-level) BPE tokenizer.
    Args:
        input_path (str):  Path to a text file with BPE tokenizer training data..
        vocab_size (int): A positive integer that defines the maximum final vocabulary size (including the initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
        special_tokens (list[str]): A list of strings to add to the vocabulary. These special tokens do not otherwise affect BPE training.
    Returns:
        tuple: A tuple containing:
            - vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes).
            - merges (list[tuple[bytes, bytes]]): A list of tuples representing the BPE merges.
    """
    # Prepare special token bytes for exclusion
    special_token_bytes = set(token.encode('utf-8') for token in special_tokens)
    vocab = init_vocab(vocab_size, special_token_bytes)    
    
    # Pretokenize input file and frequencies of words
    word_freqs = get_word_freq_mp(input_path, NUM_PROCESSES, special_tokens)

    # Count initial byte pair frequencies (weighted by word frequency)
    pair_freqs = {} # pair[bytes, bytes] -> int
    pq = PriorityQueue()
    token_sequences = {} # bytes -> list[bytes]
    pair_locations = defaultdict(set) # pair[bytes, bytes] -> set[bytes]
    for word_bytes, freq in word_freqs.items():
        byte_list = [bytes([b]) for b in word_bytes]  # Store original sequence for later updates
        token_sequences[word_bytes] = byte_list
        for i in range(len(byte_list) - 1):
            pair = (byte_list[i], byte_list[i + 1])
            pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
            pair_locations[pair].add(word_bytes)
    for pair, freq in pair_freqs.items():
        pq.put(ReverseOrder((freq, pair)))  # Max-heap

    merges = []
    next_token_id = max(vocab.keys()) + 1

    # BPE merge loop
    while len(vocab) < vocab_size:
        freq = 0
        while not pq.empty():
            # Find the most frequent byte pair
            freq, most_frequent_pair = pq.get().val
            # There are stale entries in the priority queue as we cannot remove them, skip them here instead
            if pair_freqs.get(most_frequent_pair, -1) == freq:
                break
        if freq == 0:
            break
        new_token = most_frequent_pair[0] + most_frequent_pair[1]
        vocab[next_token_id] = new_token
        merges.append(most_frequent_pair)
        next_token_id += 1

        # Update token sequences and re-count pairs
        deltas = { most_frequent_pair: -pair_freqs[most_frequent_pair] }
        for word_bytes in pair_locations[most_frequent_pair]:
            tokens = token_sequences[word_bytes]
            freq = word_freqs[word_bytes]            
            i = 0
            while i < len(tokens) - 1:
                cur_pair = (tokens[i], tokens[i + 1])
                if cur_pair == most_frequent_pair:
                    # Update left neighbor
                    if i > 0:
                        left_pair = (tokens[i - 1], tokens[i])
                        deltas[left_pair] = deltas.get(left_pair, 0) - freq
                        new_left_pair = (tokens[i - 1], new_token)
                        deltas[new_left_pair] = deltas.get(new_left_pair, 0) + freq
                        pair_locations[new_left_pair].add(word_bytes)
                    # Update right neighbor
                    if i < len(tokens) - 2:
                        right_pair = (tokens[i + 1], tokens[i + 2])
                        deltas[right_pair] = deltas.get(right_pair, 0) - freq
                        new_right_pair = (new_token, tokens[i + 2])
                        deltas[new_right_pair] = deltas.get(new_right_pair, 0) + freq
                        pair_locations[new_right_pair].add(word_bytes)
                    tokens[i] = new_token
                    del tokens[i + 1]
                    continue
                i += 1
        # Apply deltas
        for pair, delta in deltas.items():
            pair_freqs[pair] = pair_freqs.get(pair, 0) + delta
            if pair_freqs[pair] <= 0:
                del pair_freqs[pair]
                pair_locations[pair].clear()
            else:
                pq.put(ReverseOrder((pair_freqs[pair], pair)))

    return vocab, merges
