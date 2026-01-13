#!/usr/bin/env python3
"""
ActuallyOpenAI Model Training Script.

This script trains the AOAI model on text data.
Can be run locally or distributed across workers.

Usage:
    python train_model.py --data data/train.txt --epochs 10
    python train_model.py --data-url https://example.com/data.txt --epochs 5
"""

import argparse
import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import requests
from tqdm import tqdm

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from actuallyopenai.models.base_model import AOAIModel, ModelConfig
from actuallyopenai.data.tokenizer import SimpleByteTokenizer


class TextDataset(Dataset):
    """Simple text dataset for training."""
    
    def __init__(self, text: str, tokenizer: SimpleByteTokenizer, seq_length: int = 256):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        # Tokenize all text
        self.tokens = tokenizer.encode(text)
        
        # Calculate number of sequences
        self.num_sequences = max(1, (len(self.tokens) - 1) // seq_length)
        
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        start = idx * self.seq_length
        end = start + self.seq_length + 1
        
        chunk = self.tokens[start:end]
        
        # Pad if needed
        if len(chunk) < self.seq_length + 1:
            chunk = chunk + [0] * (self.seq_length + 1 - len(chunk))
        
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        
        return x, y


def get_sample_data() -> str:
    """Get sample training data."""
    
    # Sample diverse text for training
    sample_texts = [
        # Conversational
        "User: Hello, how are you?\nAssistant: I'm doing well, thank you for asking! How can I help you today?\n",
        "User: What is machine learning?\nAssistant: Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.\n",
        "User: Can you explain neural networks?\nAssistant: Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes that process information using connectionist approaches to computation.\n",
        "User: What is Python?\nAssistant: Python is a high-level, interpreted programming language known for its simplicity and readability. It's widely used in web development, data science, AI, and automation.\n",
        "User: How does deep learning work?\nAssistant: Deep learning uses multiple layers of neural networks to progressively extract higher-level features from raw input. For example, in image recognition, lower layers might identify edges, while higher layers identify faces.\n",
        
        # Technical
        "The transformer architecture revolutionized natural language processing by introducing self-attention mechanisms that allow models to process sequences in parallel.\n",
        "Distributed computing enables the processing of large datasets by splitting work across multiple machines, significantly reducing training time for large models.\n",
        "Gradient descent is an optimization algorithm that iteratively adjusts parameters to minimize a loss function by moving in the direction of steepest descent.\n",
        
        # General knowledge
        "The Earth orbits the Sun at an average distance of about 93 million miles, completing one orbit every 365.25 days.\n",
        "Water is composed of two hydrogen atoms and one oxygen atom, represented by the chemical formula H2O.\n",
        "The speed of light in a vacuum is approximately 299,792 kilometers per second.\n",
        
        # Instructions
        "To write good code, follow these principles: write clear comments, use meaningful variable names, break complex functions into smaller ones, and test your code thoroughly.\n",
        "When solving problems, start by understanding the problem completely, then break it into smaller parts, solve each part, and combine the solutions.\n",
        
        # More conversational examples
        "User: Tell me a joke.\nAssistant: Why do programmers prefer dark mode? Because light attracts bugs!\n",
        "User: What's the weather like?\nAssistant: I don't have access to real-time weather data, but I'd recommend checking a weather service for accurate information.\n",
        "User: Help me with my code.\nAssistant: I'd be happy to help! Please share the code you're working on and describe the issue you're experiencing.\n",
    ]
    
    # Repeat to create more training data
    return "\n".join(sample_texts * 50)


def download_data(url: str) -> str:
    """Download training data from URL."""
    print(f"Downloading data from {url}...")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.text


def train(
    data_path: str = None,
    data_url: str = None,
    output_path: str = "trained_model.pt",
    model_size: str = "tiny",
    epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 3e-4,
    seq_length: int = 256,
    save_every: int = 1000,
    use_sample_data: bool = True,
):
    """Train the AOAI model."""
    
    print("=" * 60)
    print("  ActuallyOpenAI Model Training")
    print("=" * 60)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load data
    print("\n[1/5] Loading data...")
    if data_path and os.path.exists(data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
    elif data_url:
        text = download_data(data_url)
    elif use_sample_data:
        print("Using built-in sample data for training...")
        text = get_sample_data()
    else:
        raise ValueError("No data source specified")
    
    print(f"Loaded {len(text):,} characters of training data")
    
    # Initialize tokenizer
    print("\n[2/5] Initializing tokenizer...")
    tokenizer = SimpleByteTokenizer()
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create dataset
    print("\n[3/5] Creating dataset...")
    dataset = TextDataset(text, tokenizer, seq_length)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False
    )
    print(f"Dataset: {len(dataset)} sequences of length {seq_length}")
    print(f"Batches per epoch: {len(dataloader)}")
    
    # Create model
    print(f"\n[4/5] Creating {model_size} model...")
    config_map = {
        "tiny": ModelConfig.tiny(),
        "small": ModelConfig.small(),
        "medium": ModelConfig.medium(),
        "large": ModelConfig.large(),
    }
    config = config_map.get(model_size, ModelConfig.tiny())
    config.vocab_size = tokenizer.vocab_size
    config.max_seq_length = seq_length
    
    model = AOAIModel(config)
    model.to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    
    # Load existing checkpoint if available
    if os.path.exists(output_path):
        try:
            checkpoint = torch.load(output_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded existing checkpoint from {output_path}")
            else:
                model.load_state_dict(checkpoint)
                print(f"Loaded existing model from {output_path}")
        except Exception as e:
            print(f"Could not load checkpoint: {e}, starting fresh")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(dataloader))
    
    # Training loop
    print(f"\n[5/5] Training for {epochs} epochs...")
    print("-" * 60)
    
    model.train()
    global_step = 0
    best_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_start = time.time()
        
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        
        for batch_idx, (x, y) in enumerate(progress):
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            logits, loss = model(x, labels=y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            # Update progress bar
            progress.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
            
            # Save checkpoint periodically
            if global_step % save_every == 0:
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'step': global_step,
                    'loss': loss.item(),
                }
                torch.save(checkpoint, output_path)
        
        # Epoch summary
        avg_loss = epoch_loss / len(dataloader)
        epoch_time = time.time() - epoch_start
        
        print(f"\nEpoch {epoch+1} complete:")
        print(f"  Average loss: {avg_loss:.4f}")
        print(f"  Time: {epoch_time:.1f}s")
        print(f"  Samples/sec: {len(dataset) / epoch_time:.1f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': {
                    'model_size': model_size,
                    'vocab_size': config.vocab_size,
                    'hidden_size': config.hidden_size,
                    'num_layers': config.num_layers,
                    'num_heads': config.num_heads,
                },
                'epoch': epoch,
                'step': global_step,
                'loss': best_loss,
            }
            torch.save(checkpoint, output_path)
            print(f"  Saved best model (loss: {best_loss:.4f})")
    
    # Final summary
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Model saved to: {output_path}")
    
    # Test generation
    print("\n" + "-" * 60)
    print("  Test Generation")
    print("-" * 60)
    
    model.eval()
    test_prompts = [
        "User: Hello!\nAssistant:",
        "User: What is AI?\nAssistant:",
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        
        with torch.no_grad():
            generated = input_tensor.clone()
            for _ in range(50):
                logits, _ = model(generated[:, -seq_length:])
                next_token_logits = logits[:, -1, :] / 0.8
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
                if next_token.item() in [0, 10]:  # EOS or newline
                    break
        
        new_tokens = generated[0, len(input_ids):].tolist()
        response = tokenizer.decode(new_tokens)
        print(f"Response: {response}")
    
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ActuallyOpenAI model")
    parser.add_argument("--data", type=str, help="Path to training data file")
    parser.add_argument("--data-url", type=str, help="URL to download training data")
    parser.add_argument("--output", type=str, default="trained_model.pt", help="Output model path")
    parser.add_argument("--model-size", type=str, default="tiny", choices=["tiny", "small", "medium", "large"])
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seq-length", type=int, default=256, help="Sequence length")
    
    args = parser.parse_args()
    
    train(
        data_path=args.data,
        data_url=args.data_url,
        output_path=args.output,
        model_size=args.model_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seq_length=args.seq_length,
    )
