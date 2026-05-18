# llm_parts.py — defines every neural-network building block used by the
# GPT-style language model trained in this project.
#
# Architecture overview (bottom-up):
#   LayerNorm          — normalises activations along the embedding dimension
#   GELU               — smooth non-linear activation function
#   FeedForward        — position-wise two-layer MLP with expand-then-contract shape
#   MultiHeadAttention — scaled dot-product self-attention split across multiple heads
#   TransformerBlock   — one full decoder layer: attention + feed-forward with residuals
#   GPTModel           — full model: token/position embeddings + N transformer blocks
import torch
from torch import nn


# -----------------------------------------------------------------------
# LayerNorm
# -----------------------------------------------------------------------
# Custom Layer Normalisation implementation (instead of using nn.LayerNorm)
# so that the normalisation logic is transparent for learning purposes.
#
# Layer norm computes, for each sample and position independently:
#   y = (x - mean) / sqrt(var + eps) * scale + shift
#
# `scale` and `shift` are learnable parameters initialised to 1 and 0
# so that the initial transformation is the identity.
# `eps` (1e-5) prevents division by zero when variance is near zero.
class LayerNorm(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        # Small constant added to the variance before taking the square root
        # to avoid numerical instability / division by zero
        self.eps = 1e-5
        # Learnable per-dimension gain; initialised to 1 (no scaling)
        self.scale = nn.Parameter(torch.ones(emb_dim))
        # Learnable per-dimension bias; initialised to 0 (no shift)
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        # Compute mean and variance along the last (embedding) dimension
        mean = x.mean(dim=-1, keepdim=True)
        # unbiased=False uses the biased (population) variance estimator
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # Normalise the input to zero mean and unit variance
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        # Apply the learnable affine transformation (scale and shift)
        return self.scale * norm_x + self.shift


# -----------------------------------------------------------------------
# GELU activation function
# -----------------------------------------------------------------------
# Gaussian Error Linear Unit — a smooth approximation of ReLU that is
# standard in modern language models (BERT, GPT-2, etc.).
#
# The approximation formula used here is:
#   GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
#
# Compared with ReLU, GELU produces non-zero gradients for negative inputs,
# which empirically leads to better training dynamics in deep transformers.
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # tanh-based polynomial approximation of the Gaussian CDF
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


# -----------------------------------------------------------------------
# FeedForward (position-wise MLP)
# -----------------------------------------------------------------------
# Each transformer block contains one position-wise feed-forward network.
# It applies the same two-layer MLP independently to every token position.
#
# Typical design: expand the embedding by 4× into a hidden layer, apply
# GELU activation, then contract back to the original embedding size.
# The 4× expansion gives the model extra capacity to learn complex patterns.
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg['emb_dim'], 4 * cfg['emb_dim']), ## Expansion
            GELU(), ## Activation
            nn.Linear(4 * cfg['emb_dim'], cfg['emb_dim']) ## Contraction
        )
    
    # Apply the MLP to every token position independently (no cross-token mixing)
    def forward(self, x):
        return self.layers(x)


# -----------------------------------------------------------------------
# MultiHeadAttention
# -----------------------------------------------------------------------
# Implements causal (autoregressive) multi-head self-attention.
#
# Key ideas:
#  - The input is linearly projected into Q, K, V matrices of size d_out.
#  - d_out is split across num_heads heads; each head attends independently
#    over a head_dim = d_out // num_heads dimensional subspace.
#  - A causal upper-triangular mask prevents each position from attending
#    to future positions (critical for autoregressive language models).
#  - All head outputs are concatenated and projected back to d_out.
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # d_out must be evenly divisible by num_heads so each head has the same size
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce projection dim to match desired output dim

        # Separate linear layers for query, key, and value projections
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        # Dropout applied to attention weights to regularise during training
        self.dropout = nn.Dropout(dropout)
        # Causal mask: an upper-triangular matrix of ones where position (i,j)
        # is 1 when j > i, meaning position i cannot attend to positions after it.
        # Registered as a buffer so it moves to the correct device with the model.
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Shape [batch_size, num_tokens, d_out]
        queries = self.W_query(x)  # Shape [batch_size, num_tokens, d_out]
        values = self.W_value(x)  # Shape [batch_size, num_tokens, d_out]

        # We implicitly split the matrix by adding the num_heads dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with causal mask
        attn_scores = queries @ keys.transpose(2, 3) # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        # Positions masked to -inf become ~0 after softmax, effectively ignored
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Scale by sqrt(head_dim) to prevent softmax saturation in high dimensions
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        # Apply dropout to attention weights for regularisation
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        # Final linear projection mixes information across heads
        context_vec = self.out_proj(context_vec)  # Optional correction

        return context_vec


# -----------------------------------------------------------------------
# TransformerBlock
# -----------------------------------------------------------------------
# One full GPT decoder layer consisting of:
#   1. Pre-norm multi-head self-attention with residual connection
#   2. Pre-norm position-wise feed-forward network with residual connection
#
# "Pre-norm" means LayerNorm is applied to the input BEFORE the sub-layer
# (as opposed to post-norm where it is applied after).  Pre-norm leads to
# more stable training gradients in deep models.
#
# Residual (skip) connections allow gradients to flow directly back through
# many layers without vanishing, enabling training of very deep networks.
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Self-attention sub-layer: input and output both have shape emb_dim
        self.att = MultiHeadAttention(
            d_in=cfg['emb_dim'],
            d_out=cfg['emb_dim'],
            context_length=cfg['context_length'],
            num_heads=cfg['num_heads'],
            dropout=cfg['drop_rate'],
            qkv_bias=cfg['qkv_bias']
        )
        # Feed-forward sub-layer applied position-wise after attention
        self.ff = FeedForward(cfg)
        # Two separate LayerNorm instances — one before attention, one before FFN
        self.norm1 = LayerNorm(cfg['emb_dim'])
        self.norm2 = LayerNorm(cfg['emb_dim'])
        # Dropout applied to the sub-layer output before adding the residual
        self.drop_shortcut = nn.Dropout(cfg['drop_rate'])
    
    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        # Normalise before passing into the attention sub-layer (pre-norm)
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_dim]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed forward block
        shortcut = x
        # Normalise before passing into the feed-forward sub-layer (pre-norm)
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


# -----------------------------------------------------------------------
# GPTModel — the top-level model class
# -----------------------------------------------------------------------
# Combines token embeddings, positional embeddings, N transformer blocks,
# a final LayerNorm, and a linear output head into a complete GPT model.
#
# cfg keys used:
#   vocab_size     — number of unique tokens in the tokenizer vocabulary
#   context_length — maximum sequence length the model can process
#   emb_dim        — dimensionality of all hidden representations
#   num_heads      — number of attention heads per transformer block
#   num_layers     — number of stacked TransformerBlock layers
#   drop_rate      — dropout probability applied throughout the model
#   qkv_bias       — whether to include bias in Q/K/V linear projections
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Token embedding: maps each token ID to a dense vector of size emb_dim
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        # Positional embedding: maps each position index to a dense vector;
        # learned positional encodings (not fixed sinusoidal ones)
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        # Dropout applied to the sum of token + positional embeddings
        self.drop_emb = nn.Dropout(cfg['drop_rate'])

        # Stack of N transformer blocks; nn.Sequential applies them in order
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg['num_layers'])]
        )

        # Final LayerNorm applied after all transformer blocks before the output head
        self.final_norm = LayerNorm(cfg['emb_dim'])
        # Linear output head projects from emb_dim to vocab_size;
        # bias=False is conventional for the output projection in GPT models
        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)
    
    # in_idx : (batch_size, seq_len) tensor of token IDs
    # Returns logits of shape (batch_size, seq_len, vocab_size) — one score
    # vector per token position over the entire vocabulary.
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        # Look up the dense embedding for every token in the batch
        tok_embds = self.tok_emb(in_idx)
        # Generate position indices [0, 1, ..., seq_len-1] on the correct device
        pos_embds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        # Add positional embeddings to token embeddings (broadcasting over batch)
        x = tok_embds + pos_embds  # Shape [ batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        # Pass through all transformer blocks sequentially
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        # Project to vocabulary size to obtain unnormalised logit scores
        logits = self.out_head(x)

        return logits


# -----------------------------------------------------------------------
# Quick smoke-test — run this file directly to verify forward pass shapes
# and that text generation produces valid token sequences.
# -----------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.tokenizer import BPETokenizer
    from src.dataloader import create_dataloader
    from src.llm_utils import generate_text

    # with open("dataset/The_Verdict.txt") as f:
    #     raw_text = f.read()
    
    # Load the tokenizer to determine the vocabulary size for the model config
    tokenizer = BPETokenizer("dataset/tokenizer.json")
    # dataloader = create_dataloader(raw_text, tokenizer, batch_size=8, max_length=4, stride=4, shuffle=False)

    # data_iter = iter(dataloader)
    # inputs, targets = next(data_iter)
    
    # Example config matching a ~124 M parameter GPT-2-scale model
    GPT_CONFIG_124M = {
        'vocab_size': len(tokenizer.str2int),  # must match tokenizer vocab
        'context_length': 1024,
        "emb_dim": 768,
        'num_heads': 12,
        "num_layers": 12,
        'drop_rate': 0.1,
        'qkv_bias': False
    }
    # batch = inputs

    # torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    # out = model(batch)
    # print(f"Input batch: {batch}")
    # print(f"Output shape: {out.shape}")
    # print(out)
    
    # num_params = sum(p.numel() for p in model.parameters())
    # print(f"Number of parameters: {num_params}")


    start_context = "Hello, I am "
    encoded = tokenizer.encode(start_context)
    print("encoded:", encoded)
    # Add batch dimension so the model receives input of shape (1, n_tokens)
    encoded_tensor = encoded.unsqueeze(0) #A
    print("encoded_tensor.shape:", encoded_tensor.shape)

    model.eval() #A
    out = generate_text(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
    )
    print("Output:", out)
    print("Output length:", len(out[0]))

    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print("Decoded text:", decoded_text)
