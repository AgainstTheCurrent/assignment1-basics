from functools import reduce
import os
from typing import BinaryIO, Iterator
from functools import reduce
import regex as re
import multiprocessing
import gc


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    desired_chunk_size: int,
    split_special_token: bytes = b"<|endoftext|>",
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = desired_chunk_size
    desired_num_chunks = max(file_size // chunk_size, desired_num_chunks)
    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def get_words(chunk: str, special_tokens: list[str], keep_special_tokens: bool = False) -> Iterator[bytes]:
    """
    Pretokenize a chunk of text and return a dictionary of pre-token counts.    
    """

    # Split chunk by special tokens
    if special_tokens:
        split_pattern = "|".join(re.escape(token) for token in special_tokens)
        split_regx = re.compile(
            f"({split_pattern})" if keep_special_tokens else split_pattern)
        parts = re.split(split_regx, chunk)
    else:
        parts = [chunk]
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    word_freq = {}
    for part in parts:
        if keep_special_tokens and part in special_tokens:
            yield part.encode("utf-8")
            continue
        for word_match in re.finditer(PAT, part):
            word = word_match.group(0).encode("utf-8")
            yield word


def get_word_freq(args: tuple[str, list[str]]) -> dict[bytes, int]:
    """
    Pretokenize a chunk of text and return a dictionary of pre-token counts.    
    """
    chunk, special_tokens = args
    word_freq = {}
    for word in get_words(chunk, special_tokens):
        word_freq[word] = word_freq.get(word, 0) + 1
    return word_freq


def merge_dicts(d1, d2, reducer_args=None):
    for token, count in d2.items():
        d1[token] = d1.get(token, 0) + count
    return d1

def pretokenize(input_path: str, num_processes: int,
                chunk_processor, reducer, processor_args, reducer_args) -> dict[bytes, int]:
    f = open(input_path, "rb")
    boundaries = find_chunk_boundaries(f, desired_num_chunks=num_processes,
                                       desired_chunk_size=10*1024*1024, # 10 MB
                                       split_special_token="<|endoftext|>".encode("utf-8"))

    chunk_idx = 0
    num_chunks = len(boundaries) - 1
    result = None
    while chunk_idx < num_chunks:
        print(f"Prepared chunk {chunk_idx+1}/{num_chunks}")
        batch_size = min(num_processes, num_chunks - chunk_idx)
        args_list = []
        for i in range(batch_size):
            start = boundaries[chunk_idx + i]
            end = boundaries[chunk_idx + i + 1]
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            args_list.append((chunk, processor_args))
        with multiprocessing.Pool(processes=batch_size) as pool:
            chunk_results = pool.map(chunk_processor, args_list)
        del args_list
        gc.collect()
        if not chunk_results:
            break
        chunk_idx += batch_size
        if result is None:
            if not chunk_results:
                return None
            result = type(chunk_results[0])()
        for chunk_result in chunk_results:
            result = reducer(result, chunk_result, reducer_args)
    f.close()
    return result


def get_word_freq_mp(input_path: str, num_processes: int, special_tokens: list[str], reducer_args = None) -> dict[bytes, int]:
    """
    Pretokenize a chunk of text and return a dictionary of pre-token counts. Use multiprocessing.
    """
    return pretokenize(input_path, num_processes, get_word_freq, merge_dicts, special_tokens, reducer_args)
