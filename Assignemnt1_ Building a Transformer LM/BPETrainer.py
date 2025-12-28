import os
import re
import regex
import pickle
import multiprocessing as mp
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, BinaryIO, Iterable
import time

class BPETrainer:
    def __init__(self, vocab_size: int = 10000, special_tokens: List[str] = None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or []
        
        # GPT-2 style regex pattern from assignment
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    def find_chunk_boundaries(
        self,
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
    ) -> List[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
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
    
    def pre_tokenize(self, text: str) -> List[str]:
        """Pre-tokenize text using GPT-2 style regex"""
        return regex.findall(self.PAT, text)
    
    def process_chunk(self, chunk_text: str) -> Dict[str, int]:
        """Process a single chunk and return word frequencies"""
        word_freqs = defaultdict(int)
        
        # Remove special tokens and split text
        if self.special_tokens:
            # Split on special tokens
            pattern = '|'.join(regex.escape(token) for token in self.special_tokens)
            parts = regex.split(f'({pattern})', chunk_text)
        else:
            parts = [chunk_text]
        
        for part in parts:
            part = part.strip()
            if not part or part in self.special_tokens:
                continue
                
            # Pre-tokenize each part
            pre_tokens = self.pre_tokenize(part)
            for token in pre_tokens:
                word_freqs[token] += 1
        
        return dict(word_freqs)
    
    def get_word_frequencies_parallel(self, input_path: str, num_processes: int = None) -> Dict[str, int]:
        """Get word frequencies using parallel processing"""
        if num_processes is None:
            num_processes = min(4, mp.cpu_count())
        
        print(f"Processing file with {num_processes} processes...")
        
        with open(input_path, "rb") as f:
            boundaries = self.find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
            print(f"Found {len(boundaries)-1} chunks")
            
            # Process chunks
            chunk_args = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                chunk_args.append(chunk)
        
        # Process chunks in parallel
        if num_processes > 1:
            with mp.Pool(num_processes) as pool:
                chunk_results = pool.map(self.process_chunk, chunk_args)
        else:
            chunk_results = [self.process_chunk(chunk) for chunk in chunk_args]
        
        # Combine results
        combined_freqs = defaultdict(int)
        for chunk_freqs in chunk_results:
            for word, freq in chunk_freqs.items():
                combined_freqs[word] += freq
        
        print(f"Found {len(combined_freqs)} unique words")
        return dict(combined_freqs)
    
    def get_alphabet(self, word_freqs: Dict[str, int]) -> set:
        """Get all unique bytes from the corpus"""
        alphabet = set()
        for word in word_freqs.keys():
            for byte in word.encode('utf-8'):
                alphabet.add(byte)
        return alphabet
    
    def split_word_to_bytes(self, word: str) -> List[bytes]:
        """Split word into individual bytes"""
        return [bytes([b]) for b in word.encode('utf-8')]
    
    def get_pairs(self, word: List[bytes]) -> set:
        """Get all adjacent pairs in a word"""
        pairs = set()
        for i in range(len(word) - 1):
            pairs.add((word[i], word[i + 1]))
        return pairs
    
    def merge_vocab(self, pair: Tuple[bytes, bytes], word_splits: Dict[str, List[bytes]]) -> Dict[str, List[bytes]]:
        """Merge the most frequent pair in vocabulary"""
        new_word_splits = {}
        
        for word in word_splits:
            new_word = []
            i = 0
            while i < len(word_splits[word]):
                if (i < len(word_splits[word]) - 1 and 
                    word_splits[word][i] == pair[0] and 
                    word_splits[word][i + 1] == pair[1]):
                    new_word.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_word.append(word_splits[word][i])
                    i += 1
            new_word_splits[word] = new_word
            
        return new_word_splits
    
    def train(self, input_path: str, num_processes: int = None) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
        """
        Train BPE tokenizer on input file
        
        Returns:
            vocab: dict[int, bytes] - mapping from token ID to bytes
            merges: list[tuple[bytes, bytes]] - list of merge operations
        """
        print("Starting BPE training...")
        start_time = time.time()
        
        # Step 1: Get word frequencies
        word_freqs = self.get_word_frequencies_parallel(input_path, num_processes)
        print("word_freqs",word_freqs)
        # Step 2: Initialize vocabulary with bytes + special tokens
        vocab = {}
        
        # Add all 256 possible bytes
        for i in range(256):
            vocab[i] = bytes([i])
        print("vocab",vocab)
        # Add special tokens
        for token in self.special_tokens:
            vocab[len(vocab)] = token.encode('utf-8')
        
        print(f"Initial vocabulary size: {len(vocab)}")
        
        # Step 3: Split all words into bytes
        print("Splitting words into bytes...")
        word_splits = {}
        for word, freq in word_freqs.items():
            word_splits[word] = self.split_word_to_bytes(word)
        
        # Step 4: Perform BPE merges
        print(f"Starting BPE merges (target vocab size: {self.vocab_size})...")
        merges = []
        
        while len(vocab) < self.vocab_size:
            # Count all pairs
            pairs = defaultdict(int)
            for word, freq in word_freqs.items():
                word_pairs = self.get_pairs(word_splits[word])
                for pair in word_pairs:
                    pairs[pair] += freq
            
            if not pairs:
                print("No more pairs to merge!")
                break
            
            # Find most frequent pair (lexicographically largest for ties)
            best_pair = max(pairs.items(), key=lambda x: (x[1], x[0]))[0]
            
            # Merge the pair
            word_splits = self.merge_vocab(best_pair, word_splits)
            
            # Add to vocabulary and merges
            new_token = best_pair[0] + best_pair[1]
            vocab[len(vocab)] = new_token
            merges.append(best_pair)
            
            if len(vocab) % 1000 == 0:
                print(f"Vocab size: {len(vocab)}, merged: {best_pair[0]} + {best_pair[1]} (freq: {pairs[best_pair]})")
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Final vocabulary size: {len(vocab)}")
        print(f"Number of merges: {len(merges)}")
        
        return vocab, merges