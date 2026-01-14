"""
Scalable Model Architecture for ActuallyOpenAI
===============================================
Designed to scale from millions to trillions of parameters.

Key Features:
- Mixture of Experts (MoE) for efficient scaling
- Flash Attention for memory efficiency
- Rotary Position Embeddings (RoPE)
- Grouped Query Attention (GQA)
- Progressive layer growth
- Checkpoint sharding for distributed training

The architecture follows modern best practices from:
- LLaMA / Mistral (RoPE, GQA, SwiGLU)
- Mixtral (MoE routing)
- DeepSeek (efficient experts)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List
from enum import Enum


class ModelScale(Enum):
    """Progressive model scales"""
    TINY = "tiny"           # 10M params - local testing
    SMALL = "small"         # 100M params - single GPU
    MEDIUM = "medium"       # 1B params - multi-GPU
    LARGE = "large"         # 7B params - distributed
    XLARGE = "xlarge"       # 70B params - cluster
    FRONTIER = "frontier"   # 400B+ params - full network


@dataclass
class ScalableConfig:
    """Configuration for scalable model"""
    # Core dimensions
    vocab_size: int = 32000
    hidden_size: int = 512
    intermediate_size: int = 2048
    num_layers: int = 8
    num_heads: int = 8
    num_kv_heads: int = 2  # For GQA (Grouped Query Attention)
    
    # MoE settings
    num_experts: int = 1  # 1 = dense, >1 = MoE
    num_experts_per_token: int = 2
    expert_capacity_factor: float = 1.25
    
    # Architecture
    max_position_embeddings: int = 8192
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    
    # Training
    dropout: float = 0.0
    attention_dropout: float = 0.0
    
    # Efficiency
    use_flash_attention: bool = True
    gradient_checkpointing: bool = False
    
    @classmethod
    def for_scale(cls, scale) -> "ScalableConfig":
        """Get config for a specific scale"""
        # Convert string to ModelScale if needed
        if isinstance(scale, str):
            scale = ModelScale(scale.lower())
        configs = {
            ModelScale.TINY: cls(
                hidden_size=256, intermediate_size=1024,
                num_layers=6, num_heads=4, num_kv_heads=2,
                num_experts=1, vocab_size=32000
            ),
            ModelScale.SMALL: cls(
                hidden_size=768, intermediate_size=3072,
                num_layers=12, num_heads=12, num_kv_heads=4,
                num_experts=1, vocab_size=32000
            ),
            ModelScale.MEDIUM: cls(
                hidden_size=2048, intermediate_size=8192,
                num_layers=24, num_heads=16, num_kv_heads=4,
                num_experts=8, num_experts_per_token=2,
                vocab_size=32000
            ),
            ModelScale.LARGE: cls(
                hidden_size=4096, intermediate_size=14336,
                num_layers=32, num_heads=32, num_kv_heads=8,
                num_experts=8, num_experts_per_token=2,
                vocab_size=32000
            ),
            ModelScale.XLARGE: cls(
                hidden_size=8192, intermediate_size=28672,
                num_layers=80, num_heads=64, num_kv_heads=8,
                num_experts=16, num_experts_per_token=2,
                vocab_size=32000, gradient_checkpointing=True
            ),
            ModelScale.FRONTIER: cls(
                hidden_size=16384, intermediate_size=53248,
                num_layers=128, num_heads=128, num_kv_heads=16,
                num_experts=64, num_experts_per_token=4,
                vocab_size=128000, gradient_checkpointing=True,
                max_position_embeddings=32768
            ),
        }
        return configs[scale]
    
    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads
    
    def estimate_params(self) -> int:
        """Estimate total parameters"""
        # Embeddings
        embed_params = self.vocab_size * self.hidden_size * 2  # input + output
        
        # Per layer
        attn_params = (
            self.hidden_size * self.hidden_size +  # Q
            self.hidden_size * (self.hidden_size // self.num_heads * self.num_kv_heads) * 2 +  # K, V
            self.hidden_size * self.hidden_size  # O
        )
        
        if self.num_experts > 1:
            # MoE: each expert has its own FFN
            ffn_params = self.num_experts * (
                self.hidden_size * self.intermediate_size * 3  # gate, up, down
            )
            ffn_params += self.hidden_size * self.num_experts  # router
        else:
            ffn_params = self.hidden_size * self.intermediate_size * 3
        
        layer_params = attn_params + ffn_params + self.hidden_size * 4  # norms
        
        total = embed_params + self.num_layers * layer_params
        return total


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""
    def __init__(self, dim: int, max_position_embeddings: int = 8192, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        self._set_cos_sin_cache(max_position_embeddings)
    
    def _set_cos_sin_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_position_embeddings:
            self._set_cos_sin_cache(seq_len)
        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len]
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """Apply rotary position embeddings to Q and K"""
    if position_ids is not None:
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
    else:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA)
    Uses fewer KV heads than Q heads for memory efficiency
    """
    def __init__(self, config: ScalableConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta
        )
        
        self.attention_dropout = nn.Dropout(config.attention_dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape to [batch, heads, seq, head_dim]
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings
        cos, sin = self.rotary_emb(value_states, seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        
        # Handle KV cache
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        past_key_value = (key_states, value_states) if use_cache else None
        
        # Repeat KV heads for GQA
        key_states = key_states.repeat_interleave(self.num_kv_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_kv_groups, dim=1)
        
        # Compute attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = self.attention_dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, past_key_value


class SwiGLU(nn.Module):
    """SwiGLU activation (used in LLaMA, Mistral)"""
    def __init__(self, config: ScalableConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ExpertLayer(nn.Module):
    """Single expert (SwiGLU FFN)"""
    def __init__(self, config: ScalableConfig):
        super().__init__()
        self.ffn = SwiGLU(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class MoELayer(nn.Module):
    """
    Mixture of Experts Layer
    Routes tokens to top-k experts for efficient scaling
    """
    def __init__(self, config: ScalableConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_token = config.num_experts_per_token
        self.hidden_size = config.hidden_size
        
        # Router
        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        
        # Experts
        self.experts = nn.ModuleList([ExpertLayer(config) for _ in range(self.num_experts)])
        
        # Load balancing loss coefficient
        self.router_aux_loss_coef = 0.01
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_size)
        
        # Get router logits and probabilities
        router_logits = self.router(hidden_states_flat)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        topk_probs, topk_indices = torch.topk(router_probs, self.num_experts_per_token, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)  # Normalize
        
        # Initialize output
        final_output = torch.zeros_like(hidden_states_flat)
        
        # Route to experts
        for i, expert in enumerate(self.experts):
            # Find tokens routed to this expert
            expert_mask = (topk_indices == i).any(dim=-1)
            if not expert_mask.any():
                continue
            
            # Get tokens for this expert
            expert_input = hidden_states_flat[expert_mask]
            expert_output = expert(expert_input)
            
            # Get weights for this expert
            expert_weights = torch.where(
                topk_indices[expert_mask] == i,
                topk_probs[expert_mask],
                torch.zeros_like(topk_probs[expert_mask])
            ).sum(dim=-1, keepdim=True)
            
            # Add weighted output
            final_output[expert_mask] += expert_weights * expert_output
        
        # Compute auxiliary load balancing loss
        aux_loss = self._compute_aux_loss(router_probs, topk_indices)
        
        return final_output.view(batch_size, seq_len, hidden_size), aux_loss
    
    def _compute_aux_loss(self, router_probs: torch.Tensor, topk_indices: torch.Tensor) -> torch.Tensor:
        """Compute load balancing auxiliary loss"""
        num_tokens = router_probs.shape[0]
        
        # Fraction of tokens routed to each expert
        expert_mask = F.one_hot(topk_indices, num_classes=self.num_experts).sum(dim=1)
        tokens_per_expert = expert_mask.float().sum(dim=0)
        fraction_tokens = tokens_per_expert / num_tokens
        
        # Mean router probability for each expert
        mean_router_prob = router_probs.mean(dim=0)
        
        # Aux loss encourages balanced routing
        aux_loss = self.num_experts * (fraction_tokens * mean_router_prob).sum()
        
        return aux_loss * self.router_aux_loss_coef


class TransformerBlock(nn.Module):
    """Single transformer block with GQA + MoE/SwiGLU"""
    def __init__(self, config: ScalableConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Use MoE or dense FFN
        if config.num_experts > 1:
            self.mlp = MoELayer(config)
            self.is_moe = True
        else:
            self.mlp = SwiGLU(config)
            self.is_moe = False
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]], torch.Tensor]:
        # Self attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states, attention_mask, position_ids, past_key_value, use_cache
        )
        hidden_states = residual + hidden_states
        
        # FFN (MoE or dense)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        if self.is_moe:
            hidden_states, aux_loss = self.mlp(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
            aux_loss = torch.tensor(0.0, device=hidden_states.device)
        
        hidden_states = residual + hidden_states
        
        return hidden_states, present_key_value, aux_loss


class ScalableAOAI(nn.Module):
    """
    Scalable ActuallyOpenAI Model
    
    Designed to scale from tiny (testing) to frontier (distributed network).
    Uses modern architecture: GQA, RoPE, SwiGLU, MoE.
    """
    
    def __init__(self, config: ScalableConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config, i) for i in range(config.num_layers)
        ])
        
        # Output
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embed_tokens.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Gradient checkpointing
        self.gradient_checkpointing = config.gradient_checkpointing
    
    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
    ) -> dict:
        batch_size, seq_len = input_ids.shape
        
        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Create causal mask
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=input_ids.device)
        
        causal_mask = self._create_causal_mask(seq_len, hidden_states.device, hidden_states.dtype)
        
        # Create position IDs
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        # Process through layers
        all_hidden_states = [] if output_hidden_states else None
        present_key_values = [] if use_cache else None
        total_aux_loss = torch.tensor(0.0, device=hidden_states.device)
        
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            
            past_key_value = past_key_values[i] if past_key_values else None
            
            if self.gradient_checkpointing and self.training:
                hidden_states, present_key_value, aux_loss = torch.utils.checkpoint.checkpoint(
                    layer, hidden_states, causal_mask, position_ids, past_key_value, use_cache
                )
            else:
                hidden_states, present_key_value, aux_loss = layer(
                    hidden_states, causal_mask, position_ids, past_key_value, use_cache
                )
            
            total_aux_loss = total_aux_loss + aux_loss
            
            if use_cache:
                present_key_values.append(present_key_value)
        
        # Final norm
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
        
        # Get logits
        logits = self.lm_head(hidden_states)
        
        return {
            "logits": logits,
            "past_key_values": present_key_values,
            "hidden_states": all_hidden_states,
            "aux_loss": total_aux_loss,
        }
    
    def _create_causal_mask(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Generate tokens autoregressively"""
        batch_size = input_ids.shape[0]
        past_key_values = None
        
        for _ in range(max_new_tokens):
            # Forward pass
            if past_key_values is None:
                outputs = self.forward(input_ids, use_cache=True)
            else:
                outputs = self.forward(input_ids[:, -1:], past_key_values=past_key_values, use_cache=True)
            
            logits = outputs["logits"][:, -1, :]
            past_key_values = outputs["past_key_values"]
            
            # Sample
            if temperature > 0:
                logits = logits / temperature
                
                # Top-k
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float("-inf")
                
                # Top-p
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float("-inf")
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)
            
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        return input_ids
    
    def get_num_params(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(scale: str = "tiny") -> ScalableAOAI:
    """Create model at specified scale"""
    scale_enum = ModelScale(scale)
    config = ScalableConfig.for_scale(scale_enum)
    model = ScalableAOAI(config)
    
    print(f"Created {scale} model:")
    print(f"  Parameters: {model.get_num_params():,}")
    print(f"  Layers: {config.num_layers}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Heads: {config.num_heads} (KV: {config.num_kv_heads})")
    if config.num_experts > 1:
        print(f"  Experts: {config.num_experts} (active: {config.num_experts_per_token})")
    
    return model


if __name__ == "__main__":
    # Test different scales
    for scale in ["tiny", "small", "medium"]:
        print(f"\n{'='*50}")
        model = create_model(scale)
        
        # Test forward pass
        x = torch.randint(0, 1000, (1, 32))
        with torch.no_grad():
            out = model(x)
        print(f"  Output shape: {out['logits'].shape}")
