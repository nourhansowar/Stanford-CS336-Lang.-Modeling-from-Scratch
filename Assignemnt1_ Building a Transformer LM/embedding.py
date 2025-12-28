import torch
import torch.nn as nn
import math
from einops import einsum

class Embedding(nn.Module):
    """
    Custom embedding layer that maps token IDs to dense vectors.
    Equivalent to nn.Embedding but implemented from scratch.
    """
    
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        
        self.num_embeddings = num_embeddings  # Vocabulary size
        self.embedding_dim = embedding_dim    # d_model
        
        # Create embedding matrix as a learnable parameter
        # Shape: (vocab_size, d_model)
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        print("self.weight",self.weight)
        print("self.earight.shape",self.weight.shape )
        # Initialize with truncated normal (mean=0, std=1)
        with torch.no_grad():
            nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Look up embeddings for given token IDs.
        
        Args:
            token_ids: Long tensor of shape (batch_size, sequence_length)
            
        Returns:
            Embeddings of shape (batch_size, sequence_length, embedding_dim)
        """
        # Simple indexing - PyTorch handles this efficiently
        return self.weight[token_ids]

# Example usage and testing
def test_embedding():
    vocab_size = 50257  # GPT-2 vocabulary size
    d_model = 768
    
    embedding = Embedding(vocab_size, d_model)
    
    # Sample token IDs (like from a tokenizer)
    token_ids = torch.tensor([
        [1, 234, 567, 890, 123, 2],    # "Hello world ..."
        [1, 345, 678, 901, 456, 2],    # "How are you ..."
        [1, 111, 222, 333, 444, 2]     # "Another sentence ..."
    ])
    
    embeddings = embedding(token_ids)
    
    print(f"Token IDs shape: {token_ids.shape}")      # torch.Size([3, 6])
    print(f"Embeddings shape: {embeddings.shape}")    # torch.Size([3, 6, 768])
    print(f"Vocabulary size: {vocab_size}")
    print(f"Embedding dimension: {d_model}")
    print("Embdedding",embeddings )
    # Verify embeddings are different for different tokens
    emb_0 = embeddings[0, 0]  # Embedding for token 1
    emb_1 = embeddings[0, 1]  # Embedding for token 234
    
    print(f"Token 1 embedding norm: {emb_0.norm():.3f}")
    print(f"Token 234 embedding norm: {emb_1.norm():.3f}")
    print(f"Embeddings are different: {not torch.allclose(emb_0, emb_1)}")
    
    return embedding

test_embedding()