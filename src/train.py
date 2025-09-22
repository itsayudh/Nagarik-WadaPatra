import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import os
import sentencepiece as spm


VOCAB_SIZE = 32000
CONTEXT_LENGTH = 512
D_MODEL = 128
NUM_HEADS = 2
NUM_LAYERS = 2
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
NUM_EPOCHS = 5
VALIDATION_SPLIT = 0.1 # 10% of the data for validation

# File paths to your tokenizer and corpus
TOKENIZER_PATH = "tokenizer/sp.model"
CORPUS_PATH = "data/processed/corpus_cleaned.txt"

# --- Model Components (reused from previous step) ---

class CausalSelfAttention(nn.Module):
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
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        return self.out_proj(attn_output)

class TransformerBlock(nn.Module):
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
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class DecoderOnlyTransformer(nn.Module):
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

        self.lm_head.weight = self.token_embedding.weight
    
    def forward(self, x):
        batch_size, seq_len = x.shape
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(torch.arange(seq_len, device=x.device))
        x = token_emb + pos_emb
        for block in self.transformer_blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits
    
# --- Dataset and Dataloader for Packed Corpus ---

class PackedCorpusDataset(Dataset):
    """
    A dataset that takes a single long sequence of token IDs and
    breaks it into fixed-length chunks for efficient training.
    """
    def __init__(self, token_ids, context_length):
        self.token_ids = token_ids
        self.context_length = context_length
        self.num_sequences = len(self.token_ids) // self.context_length
        
    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start_idx = idx * self.context_length
        end_idx = start_idx + self.context_length
        x = self.token_ids[start_idx:end_idx]
        y = self.token_ids[start_idx+1:end_idx+1]
        return x, y

def get_dataloaders(tokenizer_path, corpus_path, context_length, batch_size, val_split, device):
    """
    Loads and tokenizes the real corpus, then creates packed datasets.
    """
    print("Loading tokenizer and corpus...")
    
    # 1. Load the SentencePiece tokenizer model
    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_path)

    # 2. Read the entire corpus file
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus_text = f.read()

    # 3. Tokenize the entire corpus into a single long sequence of token IDs
    full_corpus_ids = sp.encode_as_ids(corpus_text)
    full_corpus = torch.tensor(full_corpus_ids, dtype=torch.long)
    
    print(f"Total corpus size (after tokenization): {len(full_corpus)} tokens")
    
    # Split the corpus into train and validation sets
    split_idx = int(len(full_corpus) * (1 - val_split))
    train_corpus = full_corpus[:split_idx]
    val_corpus = full_corpus[split_idx:]
    
    print(f"Train corpus size: {len(train_corpus)} tokens")
    print(f"Validation corpus size: {len(val_corpus)} tokens")

    train_dataset = PackedCorpusDataset(train_corpus, context_length)
    val_dataset = PackedCorpusDataset(val_corpus, context_length)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_dataloader, val_dataloader

# --- Training and Validation Functions ---

def validate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data, targets in dataloader:
            data, targets = data.to(device), targets.to(device)
            logits = model(data)
            loss = loss_fn(logits.view(-1, logits.shape[-1]), targets.view(-1))
            total_loss += loss.item() * data.size(0)
    avg_loss = total_loss / len(dataloader.dataset)
    perplexity = torch.exp(torch.tensor(avg_loss))
    return avg_loss, perplexity

def train_model(model, train_dataloader, val_dataloader, num_epochs, lr, device):
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_dataloader) * num_epochs)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch_idx, (data, targets) in enumerate(train_dataloader):
            data, targets = data.to(device), targets.to(device)
            logits = model(data)
            loss = loss_fn(logits.view(-1, logits.shape[-1]), targets.view(-1))
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Step {batch_idx+1}/{len(train_dataloader)}, Train Loss: {loss.item():.4f}")

        avg_val_loss, val_perplexity = validate(model, val_dataloader, loss_fn, device)
        print(f"\n--- Epoch {epoch+1} Complete ---")
        print(f"Average Training Loss: {total_loss / len(train_dataloader):.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Perplexity: {val_perplexity:.2f}")
        print("----------------------------\n")

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = DecoderOnlyTransformer(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS
    ).to(device)

    train_dataloader, val_dataloader = get_dataloaders(
        TOKENIZER_PATH, CORPUS_PATH, CONTEXT_LENGTH, BATCH_SIZE, VALIDATION_SPLIT, device
    )

    train_model(model, train_dataloader, val_dataloader, NUM_EPOCHS, LEARNING_RATE, device)
