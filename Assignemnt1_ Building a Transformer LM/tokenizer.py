import regex
import pickle
from typing import Dict, List, Tuple, Iterator, Iterable

class Tokenizer:
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: List[str] = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        
        # Create reverse mappings
        self.token_to_id = {token: id for id, token in vocab.items()}
        self.merge_order = {merge: i for i, merge in enumerate(merges)}
        
        # Add special tokens to vocab if not present
        for token in self.special_tokens:
            token_bytes = token.encode('utf-8')
            if token_bytes not in self.token_to_id:
                new_id = len(self.vocab)
                self.vocab[new_id] = token_bytes
                self.token_to_id[token_bytes] = new_id
        
        # GPT-2 style regex pattern
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: List[str] = None):
        """Load tokenizer from serialized files"""
        # Load vocab
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)
        
        # Load merges  
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)
        
        return cls(vocab, merges, special_tokens)
    
    def pre_tokenize(self, text: str) -> List[str]:
        """Pre-tokenize text using regex"""
        return regex.findall(self.PAT, text)
    
    def handle_special_tokens(self, text: str) -> List[str]:
        """Split text on special tokens"""
        if not self.special_tokens:
            return [text]
        
        # Create pattern for special tokens
        pattern = '|'.join(regex.escape(token) for token in self.special_tokens)
        parts = regex.split(f'({pattern})', text)
        
        # Filter out empty strings
        return [part for part in parts if part]
    
    def apply_merges(self, tokens: List[bytes]) -> List[bytes]:
        """Apply BPE merges to token list"""
        while len(tokens) > 1:
            # Find best merge to apply
            best_merge = None
            best_pos = -1
            best_order = float('inf')
            
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merge_order and self.merge_order[pair] < best_order:
                    best_merge = pair
                    best_pos = i
                    best_order = self.merge_order[pair]
            
            if best_merge is None:
                break
            
            # Apply the merge
            new_tokens = (tokens[:best_pos] + 
                         [best_merge[0] + best_merge[1]] + 
                         tokens[best_pos + 2:])
            tokens = new_tokens
        
        return tokens
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        token_ids = []
        
        # Handle special tokens
        parts = self.handle_special_tokens(text)
        
        for part in parts:
            if part in self.special_tokens:
                # This is a special token
                token_bytes = part.encode('utf-8')
                token_ids.append(self.token_to_id[token_bytes])
            else:
                # Regular text - pre-tokenize and apply BPE
                pre_tokens = self.pre_tokenize(part)
                
                for pre_token in pre_tokens:
                    # Convert to bytes
                    tokens = [bytes([b]) for b in pre_token.encode('utf-8')]
                    
                    # Apply merges
                    tokens = self.apply_merges(tokens)
                    
                    # Convert to IDs
                    for token in tokens:
                        if token in self.token_to_id:
                            token_ids.append(self.token_to_id[token])
                        else:
                            # Fallback to individual bytes if token not found
                            for b in token:
                                token_ids.append(b)
        
        return token_ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Memory-efficient encoding of large text streams"""
        for text_chunk in iterable:
            for token_id in self.encode(text_chunk):
                yield token_id
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text"""
        tokens = []
        for id in ids:
            if id in self.vocab:
                tokens.append(self.vocab[id])
            else:
                # Handle unknown IDs gracefully
                tokens.append(b'')  # or some replacement token
        
        # Concatenate all byte sequences
        byte_string = b''.join(tokens)
        
        # Decode to string with error handling
        return byte_string.decode('utf-8', errors='replace')