"""
Real Model Inference for ActuallyOpenAI.

This module provides actual model inference using trained models.
"""

import os
import torch
import torch.nn.functional as F
from typing import Optional, List, Generator
from pathlib import Path
import structlog

# Import our model architecture
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from actuallyopenai.models.base_model import AOAIModel, ModelConfig
from actuallyopenai.data.tokenizer import SimpleByteTokenizer

logger = structlog.get_logger()


class ModelInference:
    """Real model inference with the trained AOAI model."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.model: Optional[AOAIModel] = None
        self.tokenizer: Optional[SimpleByteTokenizer] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_loaded = False
        self._initialized = True
        
    def load_model(self, model_path: Optional[str] = None, config_size: str = "tiny"):
        """Load the model from checkpoint or create new."""
        
        # Initialize tokenizer
        self.tokenizer = SimpleByteTokenizer()
        
        # Get model config
        config_map = {
            "tiny": ModelConfig.tiny(),
            "small": ModelConfig.small(),
            "medium": ModelConfig.medium(),
            "large": ModelConfig.large(),
        }
        config = config_map.get(config_size, ModelConfig.tiny())
        config.vocab_size = self.tokenizer.vocab_size
        
        # Create model
        self.model = AOAIModel(config)
        
        # Try to load checkpoint
        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                logger.info(f"Loaded model from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}, using fresh model")
        else:
            # Check default paths
            default_paths = [
                "trained_model.pt",
                "checkpoints/latest.pt",
                "models/aoai-1.pt",
            ]
            for path in default_paths:
                if os.path.exists(path):
                    try:
                        checkpoint = torch.load(path, map_location=self.device)
                        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                            self.model.load_state_dict(checkpoint['model_state_dict'])
                        else:
                            self.model.load_state_dict(checkpoint)
                        logger.info(f"Loaded model from {path}")
                        break
                    except:
                        continue
        
        self.model.to(self.device)
        self.model.eval()
        self.model_loaded = True
        
        param_count = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model loaded: {param_count:,} parameters on {self.device}")
        
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
    ) -> str:
        """Generate text from a prompt."""
        
        if not self.model_loaded:
            self.load_model()
        
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        
        # Generate tokens
        generated = input_tensor.clone()
        
        for _ in range(max_new_tokens):
            # Get model predictions
            if generated.size(1) > self.model.config.max_seq_length:
                context = generated[:, -self.model.config.max_seq_length:]
            else:
                context = generated
                
            logits, _ = self.model(context)
            next_token_logits = logits[:, -1, :]
            
            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample or greedy
            if do_sample and temperature > 0:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for EOS (byte 0 or 255)
            if next_token.item() == 0 or next_token.item() == 255:
                break
        
        # Decode generated text
        generated_ids = generated[0].tolist()
        new_ids = generated_ids[len(input_ids):]
        
        try:
            generated_text = self.tokenizer.decode(new_ids)
        except:
            generated_text = ""
        
        return generated_text
    
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        """Generate text token by token (streaming)."""
        
        if not self.model_loaded:
            self.load_model()
        
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        generated = input_tensor.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                if generated.size(1) > self.model.config.max_seq_length:
                    context = generated[:, -self.model.config.max_seq_length:]
                else:
                    context = generated
                    
                logits, _ = self.model(context)
                next_token_logits = logits[:, -1, :] / max(temperature, 0.1)
                
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated = torch.cat([generated, next_token], dim=1)
                
                # Decode just this token
                try:
                    token_text = self.tokenizer.decode([next_token.item()])
                    if token_text:
                        yield token_text
                except Exception as e:
                    logger.debug(f"Failed to decode token {next_token.item()}: {e}")
                
                if next_token.item() == 0 or next_token.item() == 255:
                    break
    
    def get_embedding(self, text: str) -> List[float]:
        """Get text embedding from the model."""
        
        if not self.model_loaded:
            self.load_model()
        
        input_ids = self.tokenizer.encode(text)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            # Get hidden states
            hidden_states = self.model.embed_tokens(input_tensor)
            
            # Apply transformer layers
            for layer in self.model.layers:
                hidden_states = layer(hidden_states)
            
            # Mean pooling
            embedding = hidden_states.mean(dim=1).squeeze()
            
            # Normalize
            embedding = F.normalize(embedding, dim=0)
            
            # Pad or truncate to 1536 dimensions (OpenAI compatibility)
            if embedding.size(0) < 1536:
                padding = torch.zeros(1536 - embedding.size(0), device=self.device)
                embedding = torch.cat([embedding, padding])
            else:
                embedding = embedding[:1536]
        
        return embedding.cpu().tolist()


# Global inference instance
inference = ModelInference()


def get_inference() -> ModelInference:
    """Get the global inference instance."""
    return inference
