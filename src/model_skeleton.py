import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sentencepiece as spm

# --- Configuration ---
# You can change these to test different configurations
# Based on Candidate Brief constraints (e.g., <= 50M params)
# Tiny model for the toy overfit proof
VOCAB_SIZE = 32000
CONTEXT_LENGTH = 512
D_MODEL = 128
NUM_HEADS = 2
NUM_LAYERS = 2
TOY_DATA_SIZE = 50
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
NUM_EPOCHS = 200

# --- Helper Functions and Classes ---

class SimpleTokenizer:
    """A dummy tokenizer for the toy overfit proof."""
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def encode(self, text):
        return [i for i in range(len(text))]

    def decode(self, ids):
        return [str(i) for i in ids]

class ToyDataset(Dataset):
    """A simple dataset to prove the model can overfit."""
    def __init__(self, vocab_size, context_length, size):
        self.vocab_size = vocab_size
        self.context_length = context_length
        # Create a single, repeating sequence for overfitting
        self.data = torch.randint(0, vocab_size, (context_length,), dtype=torch.long)
        self.size = size
    
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # We return a single repeating sequence and its target
        x = self.data[:-1]
        y = self.data[1:]
        return x, y

# --- Model Components ---

class CausalSelfAttention(nn.Module):
    """
    A masked self-attention module to ensure a decoder-only model.
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv_proj(x).view(batch_size, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Causal mask to prevent attending to future tokens
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)

        return self.out_proj(attn_output)

class TransformerBlock(nn.Module):
    """
    A single decoder-only transformer block.
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        # Apply attention with skip connection and normalization
        x = x + self.attn(self.norm1(x))
        # Apply feed-forward network with skip connection and normalization
        x = x + self.ffn(self.norm2(x))
        return x

class DecoderOnlyTransformer(nn.Module):
    """
    The main decoder-only Transformer model.
    """
    def __init__(self, vocab_size, context_length, d_model, num_heads, num_layers):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_length, d_model)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.context_length = context_length

        # Share weights between token embedding and LM head for efficiency
        self.lm_head.weight = self.token_embedding.weight
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        batch_size, seq_len = x.shape
        
        # Token and positional embeddings
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(torch.arange(seq_len, device=x.device))
        
        x = token_emb + pos_emb
        
        # Pass through the transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
            
        x = self.norm(x)
        logits = self.lm_head(x) # (batch_size, seq_len, vocab_size)
        
        return logits

def train_toy_model(model, dataloader, num_epochs, lr):
    """
    Trains the model on a small dataset to prove it can overfit.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(dataloader):
            data, targets = data.to(device), targets.to(device)
            
            # Forward pass
            logits = model(data)
            
            # Reshape for loss calculation
            logits = logits.view(-1, logits.shape[-1])
            targets = targets.view(-1)
            
            loss = loss_fn(logits, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    print("Toy overfit proof complete.")
    print(f"Final Loss: {loss.item():.4f}")

    # Check if the loss is low enough to prove overfitting
    if loss.item() < 0.01:
        print("\nSUCCESS: The model has successfully overfit the toy dataset. The core architecture is working correctly!")
    else:
        print("\nFAILURE: The model failed to overfit. Please check your model implementation.")

if __name__ == "__main__":
    # Set a fixed seed for reproducibility as per the brief
    torch.manual_seed(42)
    np.random.seed(42)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize the model and dataset
    toy_dataset = ToyDataset(VOCAB_SIZE, CONTEXT_LENGTH, TOY_DATA_SIZE)
    toy_dataloader = DataLoader(toy_dataset, batch_size=BATCH_SIZE)

    model = DecoderOnlyTransformer(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total model parameters: {total_params / 1e6:.2f}M")
    
    # Train the model on the toy dataset
    train_toy_model(model, toy_dataloader, NUM_EPOCHS, LEARNING_RATE)
