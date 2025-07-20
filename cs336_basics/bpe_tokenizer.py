from typing import BinaryIO, Iterable, Iterator
from cs336_basics.pretokenization import get_words, pretokenize
import pickle
import os
import numpy as np

NUM_PROCESSES = os.cpu_count() or 2

class BPETokenizer(object):
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] = None):
        """
        Construct a tokenizer from a given vocabulary, list of merges, and 
        (optionally) a list of special tokens.
        """
        self.vocab = vocab
        self.token_to_id = {t: i for i, t in vocab.items()}
        self.merges = {merge: i for i, merge in enumerate(merges)}
        self.special_tokens = sorted(special_tokens or [], reverse=True)

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens=None):
        """
        Class method that constructs and returns a Tokenizer from a serialized vocabulary
        and list of merges (in the same format that your BPE training code output) and
        (optionally) a list of special tokens.
        """
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode_word(self, word: bytes) -> list[int]:
        token_sequence = [bytes([b]) for b in word]
        while True:
            rank = None
            merge = None
            for i in range(len(token_sequence) - 1):
                pair = (token_sequence[i], token_sequence[i + 1])
                if pair in self.merges:
                    new_rank = self.merges[pair]
                    if rank is None or new_rank < rank:
                        rank = new_rank
                        merge = i
            if merge is None:
                return [self.token_to_id[tok] for tok in token_sequence]
            token_sequence[merge] = token_sequence[merge] + token_sequence[merge + 1]
            del token_sequence[merge + 1]

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of tokenIDs.
        """
        token_ids = []
        for word in get_words(text, self.special_tokens, keep_special_tokens=True):
            if word in self.token_to_id:
                token_ids.append(self.token_to_id[word])
            else:
                token_ids.extend(self.encode_word(word))
        return token_ids


    @staticmethod
    def encode_(args) -> list[int]:
        text, bpe_tokenizer = args
        return bpe_tokenizer.encode(text)


    @staticmethod
    def write_token_ids(token_ids1: Iterable[int], token_ids2: Iterable[int], output_file: BinaryIO):
        np.array(token_ids2, dtype=np.uint16).tofile(output_file)

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily
        yields tokenIDs. This is required for memory-efficient tokenization of large files that
        we cannot directly load into memory.
        """
        for text in iterable:
            for token_ids in self.encode(text):
                yield token_ids

    def encode_file(self, input_path: str, output_path: str):
        """
        Given a path to a text file, return a list of tokenIDs corresponding to the
        tokenization of the entire file.
        """
        with open(output_path, "wb") as f:
            pretokenize(input_path, NUM_PROCESSES, self.encode_, self.write_token_ids, self, f)

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of tokenIDs into text
        """
        tokens = [self.vocab[i] for i in ids]
        # Join bytes and decode
        token_bytes = b"".join(tokens)
        return token_bytes.decode("utf-8", errors="replace")
