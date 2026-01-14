"""
BPE Tokenizer for ActuallyOpenAI
================================
Proper tokenization is critical for model performance.

This implements Byte-Pair Encoding (BPE) similar to GPT-2/GPT-4.
- Handles arbitrary text (including code, math, Unicode)
- Efficient vocabulary utilization
- Trainable on custom data
"""

import os
import json
import regex as re
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from collections import Counter
import logging

logger = logging.getLogger("AOAI-Tokenizer")


class BPETokenizer:
    """
    Byte-Pair Encoding Tokenizer
    
    Similar to GPT-2/GPT-4 tokenization.
    Can be trained on custom data or load pre-trained vocab.
    """
    
    # GPT-4 style pattern for splitting text
    PAT_STR = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    
    # Special tokens
    SPECIAL_TOKENS = {
        "<|pad|>": 0,
        "<|unk|>": 1,
        "<|bos|>": 2,
        "<|eos|>": 3,
        "<|sep|>": 4,
        "<|mask|>": 5,
        # Chat tokens
        "<|im_start|>": 6,
        "<|im_end|>": 7,
        "<|user|>": 8,
        "<|assistant|>": 9,
        "<|system|>": 10,
    }
    
    def __init__(
        self,
        vocab_size: int = 32000,
        special_tokens: Optional[Dict[str, int]] = None,
    ):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or self.SPECIAL_TOKENS.copy()
        
        # Initialize with byte-level tokens
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        
        # Compile regex pattern
        try:
            self.pattern = re.compile(self.PAT_STR)
        except:
            # Fallback for systems without regex library
            self.pattern = None
        
        self._initialize_byte_vocab()
    
    def _initialize_byte_vocab(self):
        """Initialize vocabulary with byte-level tokens"""
        # Start with special tokens
        for token, idx in self.special_tokens.items():
            self.vocab[token] = idx
        
        # Add byte tokens (256 possible bytes)
        start_idx = len(self.special_tokens)
        for i in range(256):
            token = bytes([i]).decode('utf-8', errors='replace')
            if token not in self.vocab:
                self.vocab[f"<|byte_{i}|>"] = start_idx + i
        
        self._build_inverse_vocab()
    
    def _build_inverse_vocab(self):
        """Build inverse vocabulary mapping"""
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
    
    def train(self, texts: List[str], min_frequency: int = 2):
        """
        Train BPE on a corpus of text.
        
        This learns the merge rules that define the tokenizer.
        """
        logger.info(f"Training BPE tokenizer on {len(texts)} texts...")
        
        # Convert texts to byte sequences
        word_freqs = Counter()
        for text in texts:
            words = self._split_text(text)
            for word in words:
                # Convert to tuple of bytes
                byte_word = tuple(bytes(word, 'utf-8'))
                word_freqs[byte_word] += 1
        
        # Initialize vocabulary with characters
        vocab = set()
        for word in word_freqs:
            for byte in word:
                vocab.add(bytes([byte]).decode('utf-8', errors='replace'))
        
        # Learn merges until we hit vocab size
        current_vocab_size = len(self.special_tokens) + len(vocab)
        
        while current_vocab_size < self.vocab_size:
            # Count pairs
            pair_freqs = Counter()
            for word, freq in word_freqs.items():
                if len(word) < 2:
                    continue
                for i in range(len(word) - 1):
                    pair = (word[i], word[i + 1])
                    pair_freqs[pair] += freq
            
            if not pair_freqs:
                break
            
            # Find most frequent pair
            best_pair = max(pair_freqs, key=pair_freqs.get)
            if pair_freqs[best_pair] < min_frequency:
                break
            
            # Merge best pair
            self.merges.append(best_pair)
            
            # Update word frequencies with merged pair
            new_word_freqs = Counter()
            for word, freq in word_freqs.items():
                new_word = self._apply_merge(word, best_pair)
                new_word_freqs[new_word] += freq
            
            word_freqs = new_word_freqs
            current_vocab_size += 1
            
            if len(self.merges) % 1000 == 0:
                logger.info(f"Learned {len(self.merges)} merges, vocab size: {current_vocab_size}")
        
        # Build final vocabulary
        self._build_vocab_from_merges()
        logger.info(f"Training complete. Final vocab size: {len(self.vocab)}")
    
    def _split_text(self, text: str) -> List[str]:
        """Split text into words using GPT-4 style pattern"""
        if self.pattern:
            return self.pattern.findall(text)
        else:
            # Simple fallback
            return text.split()
    
    def _apply_merge(self, word: tuple, pair: Tuple[int, int]) -> tuple:
        """Apply a merge rule to a word"""
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                new_word.append(pair[0] * 256 + pair[1])  # Combined token
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return tuple(new_word)
    
    def _build_vocab_from_merges(self):
        """Build vocabulary from learned merges"""
        # Add merged tokens
        for i, (a, b) in enumerate(self.merges):
            # Create token string
            a_str = bytes([a]).decode('utf-8', errors='replace') if isinstance(a, int) and a < 256 else str(a)
            b_str = bytes([b]).decode('utf-8', errors='replace') if isinstance(b, int) and b < 256 else str(b)
            token = a_str + b_str
            
            idx = len(self.special_tokens) + 256 + i
            self.vocab[token] = idx
        
        self._build_inverse_vocab()
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs"""
        if add_special_tokens:
            tokens = [self.special_tokens["<|bos|>"]]
        else:
            tokens = []
        
        words = self._split_text(text)
        
        for word in words:
            word_tokens = self._encode_word(word)
            tokens.extend(word_tokens)
        
        if add_special_tokens:
            tokens.append(self.special_tokens["<|eos|>"])
        
        return tokens
    
    def _encode_word(self, word: str) -> List[int]:
        """Encode a single word to token IDs"""
        # Convert to bytes
        word_bytes = list(bytes(word, 'utf-8'))
        
        # Apply merges
        while len(word_bytes) > 1:
            # Find best merge
            best_merge = None
            best_idx = len(self.merges)
            
            for i in range(len(word_bytes) - 1):
                pair = (word_bytes[i], word_bytes[i + 1])
                if pair in self.merges:
                    merge_idx = self.merges.index(pair)
                    if merge_idx < best_idx:
                        best_idx = merge_idx
                        best_merge = (i, pair)
            
            if best_merge is None:
                break
            
            # Apply merge
            i, pair = best_merge
            merged = pair[0] * 256 + pair[1]
            word_bytes = word_bytes[:i] + [merged] + word_bytes[i+2:]
        
        # Convert to token IDs
        tokens = []
        for b in word_bytes:
            if b < 256:
                token = f"<|byte_{b}|>"
            else:
                # Find in vocab
                token = self.inverse_vocab.get(b, "<|unk|>")
            
            tokens.append(self.vocab.get(token, self.special_tokens["<|unk|>"]))
        
        return tokens
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text"""
        tokens = []
        
        for idx in token_ids:
            token = self.inverse_vocab.get(idx, "")
            
            if skip_special_tokens and token in self.special_tokens:
                continue
            
            # Handle byte tokens
            if token.startswith("<|byte_") and token.endswith("|>"):
                try:
                    byte_val = int(token[7:-2])
                    token = bytes([byte_val]).decode('utf-8', errors='replace')
                except:
                    pass
            
            tokens.append(token)
        
        return ''.join(tokens)
    
    def save(self, path: str):
        """Save tokenizer to file"""
        data = {
            "vocab_size": self.vocab_size,
            "vocab": self.vocab,
            "merges": [(a, b) for a, b in self.merges],
            "special_tokens": self.special_tokens,
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved tokenizer to {path}")
    
    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Load tokenizer from file"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tokenizer = cls(
            vocab_size=data["vocab_size"],
            special_tokens=data["special_tokens"],
        )
        tokenizer.vocab = data["vocab"]
        tokenizer.merges = [tuple(m) for m in data["merges"]]
        tokenizer._build_inverse_vocab()
        
        logger.info(f"Loaded tokenizer from {path}")
        return tokenizer
    
    @property
    def pad_token_id(self) -> int:
        return self.special_tokens["<|pad|>"]
    
    @property
    def eos_token_id(self) -> int:
        return self.special_tokens["<|eos|>"]
    
    @property
    def bos_token_id(self) -> int:
        return self.special_tokens["<|bos|>"]


class ChatFormatter:
    """
    Format chat conversations for training/inference.
    
    Uses chat template similar to ChatML.
    """
    
    def __init__(self, tokenizer: BPETokenizer):
        self.tokenizer = tokenizer
    
    def format_chat(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """Format a chat conversation"""
        formatted = ""
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                formatted += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        
        if add_generation_prompt:
            formatted += "<|im_start|>assistant\n"
        
        return formatted
    
    def encode_chat(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> List[int]:
        """Encode a chat conversation to token IDs"""
        formatted = self.format_chat(messages, add_generation_prompt)
        return self.tokenizer.encode(formatted, add_special_tokens=False)


# Pre-built tokenizer for common use
def get_default_tokenizer() -> BPETokenizer:
    """Get or create default tokenizer"""
    default_path = Path(__file__).parent / "default_tokenizer.json"
    
    if default_path.exists():
        return BPETokenizer.load(str(default_path))
    else:
        # Create basic tokenizer
        tokenizer = BPETokenizer(vocab_size=32000)
        return tokenizer


if __name__ == "__main__":
    # Test tokenizer
    tokenizer = BPETokenizer(vocab_size=1000)
    
    # Train on sample data
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Neural networks are inspired by biological neurons.",
        "Deep learning has revolutionized computer vision.",
        "Natural language processing enables machines to understand text.",
    ] * 100
    
    tokenizer.train(sample_texts)
    
    # Test encoding/decoding
    test_text = "Machine learning is amazing!"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"Original: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    
    # Test chat formatting
    formatter = ChatFormatter(tokenizer)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is AI?"},
    ]
    
    formatted = formatter.format_chat(messages)
    print(f"\nChat format:\n{formatted}")
