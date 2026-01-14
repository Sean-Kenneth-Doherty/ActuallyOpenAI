"""
Continuous Growth Model
=======================
AI that grows continuously rather than in discrete steps.

Instead of TINY → SMALL → MEDIUM → LARGE jumps,
the model expands smoothly:
- Dimensions grow gradually
- Layers added one at a time
- Experts added progressively
- Knowledge transferred continuously

The AI is always at its maximum capacity for available compute.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
import copy
import time

logger = logging.getLogger("AOAI-ContinuousGrowth")


@dataclass
class GrowthState:
    """Tracks the continuous growth state of the model"""
    # Current dimensions
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 4
    num_kv_heads: int = 2
    intermediate_size: int = 1024
    num_experts: int = 1
    
    # Growth history
    total_tokens_trained: int = 0
    growth_events: List[Dict[str, Any]] = field(default_factory=list)
    
    # Growth thresholds (tokens needed for each growth)
    tokens_per_hidden_growth: int = 50_000_000      # 50M tokens to grow hidden by 64
    tokens_per_layer_growth: int = 200_000_000      # 200M tokens to add a layer
    tokens_per_expert_growth: int = 500_000_000     # 500M tokens to add an expert
    
    # Growth increments
    hidden_growth_increment: int = 64
    
    # Limits (based on available compute)
    max_hidden_size: int = 16384
    max_layers: int = 128
    max_experts: int = 64
    
    def should_grow_hidden(self) -> bool:
        """Check if hidden size should grow"""
        if self.hidden_size >= self.max_hidden_size:
            return False
        threshold = (self.hidden_size // self.hidden_growth_increment) * self.tokens_per_hidden_growth
        return self.total_tokens_trained >= threshold
    
    def should_grow_layer(self) -> bool:
        """Check if we should add a layer"""
        if self.num_layers >= self.max_layers:
            return False
        threshold = self.num_layers * self.tokens_per_layer_growth
        return self.total_tokens_trained >= threshold
    
    def should_grow_expert(self) -> bool:
        """Check if we should add an expert"""
        if self.num_experts >= self.max_experts:
            return False
        # Only add experts after reaching certain size
        if self.hidden_size < 1024 or self.num_layers < 12:
            return False
        threshold = self.num_experts * self.tokens_per_expert_growth
        return self.total_tokens_trained >= threshold


class GrowableEmbedding(nn.Module):
    """Embedding layer that can grow its hidden dimension"""
    
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.randn(vocab_size, hidden_size) * 0.02)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_ids, self.weight)
    
    def grow_hidden(self, new_hidden_size: int):
        """Expand hidden dimension, preserving existing weights"""
        if new_hidden_size <= self.hidden_size:
            return
        
        old_weight = self.weight.data
        new_weight = torch.randn(self.vocab_size, new_hidden_size, device=old_weight.device) * 0.02
        new_weight[:, :self.hidden_size] = old_weight
        
        self.weight = nn.Parameter(new_weight)
        self.hidden_size = new_hidden_size
        
        logger.info(f"Embedding grown: {old_weight.shape[1]} → {new_hidden_size}")


class GrowableRMSNorm(nn.Module):
    """RMSNorm that can grow"""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x
    
    def grow_hidden(self, new_hidden_size: int):
        """Expand hidden dimension"""
        if new_hidden_size <= self.hidden_size:
            return
        
        old_weight = self.weight.data
        new_weight = torch.ones(new_hidden_size, device=old_weight.device)
        new_weight[:self.hidden_size] = old_weight
        
        self.weight = nn.Parameter(new_weight)
        self.hidden_size = new_hidden_size


class GrowableLinear(nn.Module):
    """Linear layer that can grow input and output dimensions"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * (2 / (in_features + out_features)) ** 0.5)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)
    
    def grow_input(self, new_in_features: int):
        """Expand input dimension"""
        if new_in_features <= self.in_features:
            return
        
        old_weight = self.weight.data
        new_weight = torch.zeros(self.out_features, new_in_features, device=old_weight.device)
        new_weight[:, :self.in_features] = old_weight
        # Initialize new weights with small random values
        new_weight[:, self.in_features:] = torch.randn(
            self.out_features, new_in_features - self.in_features, device=old_weight.device
        ) * 0.01
        
        self.weight = nn.Parameter(new_weight)
        self.in_features = new_in_features
    
    def grow_output(self, new_out_features: int):
        """Expand output dimension"""
        if new_out_features <= self.out_features:
            return
        
        old_weight = self.weight.data
        new_weight = torch.zeros(new_out_features, self.in_features, device=old_weight.device)
        new_weight[:self.out_features, :] = old_weight
        new_weight[self.out_features:, :] = torch.randn(
            new_out_features - self.out_features, self.in_features, device=old_weight.device
        ) * 0.01
        
        self.weight = nn.Parameter(new_weight)
        
        if self.bias is not None:
            old_bias = self.bias.data
            new_bias = torch.zeros(new_out_features, device=old_bias.device)
            new_bias[:self.out_features] = old_bias
            self.bias = nn.Parameter(new_bias)
        
        self.out_features = new_out_features
    
    def grow_both(self, new_in_features: int, new_out_features: int):
        """Expand both dimensions"""
        self.grow_input(new_in_features)
        self.grow_output(new_out_features)


class GrowableAttention(nn.Module):
    """Attention that can grow dimensions and heads"""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int = 8192,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        
        self.q_proj = GrowableLinear(hidden_size, num_heads * self.head_dim)
        self.k_proj = GrowableLinear(hidden_size, num_kv_heads * self.head_dim)
        self.v_proj = GrowableLinear(hidden_size, num_kv_heads * self.head_dim)
        self.o_proj = GrowableLinear(num_heads * self.head_dim, hidden_size)
        
        # RoPE
        self._init_rope()
    
    def _init_rope(self):
        """Initialize rotary position embeddings"""
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq)
    
    def _apply_rope(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Apply rotary position embeddings"""
        seq_len = x.shape[2]
        
        # Handle head_dim mismatch after growth
        current_head_dim = x.shape[-1]
        half_dim = current_head_dim // 2
        inv_freq = self.inv_freq[:half_dim]
        
        freqs = torch.einsum("bi,j->bij", position_ids.float(), inv_freq)
        # freqs shape: [batch, seq_len, half_dim]
        
        cos = freqs.cos().unsqueeze(1)  # [batch, 1, seq_len, half_dim]
        sin = freqs.sin().unsqueeze(1)  # [batch, 1, seq_len, half_dim]
        
        # x shape: [batch, heads, seq_len, head_dim]
        x1 = x[..., :half_dim]  # [batch, heads, seq_len, half_dim]
        x2 = x[..., half_dim:]  # [batch, heads, seq_len, half_dim]
        
        # cos/sin need to broadcast properly
        rotated = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        return rotated
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Position IDs
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        
        # Apply RoPE
        q = self._apply_rope(q, position_ids)
        k = self._apply_rope(k, position_ids)
        
        # Expand KV for GQA
        if self.num_kv_heads < self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)
        
        # Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn_output)
    
    def grow_hidden(self, new_hidden_size: int):
        """Grow hidden dimension (requires updating projections)"""
        if new_hidden_size <= self.hidden_size:
            return
        
        # Calculate new head dim
        new_head_dim = new_hidden_size // self.num_heads
        
        # Grow projections
        self.q_proj.grow_both(new_hidden_size, self.num_heads * new_head_dim)
        self.k_proj.grow_both(new_hidden_size, self.num_kv_heads * new_head_dim)
        self.v_proj.grow_both(new_hidden_size, self.num_kv_heads * new_head_dim)
        self.o_proj.grow_both(self.num_heads * new_head_dim, new_hidden_size)
        
        self.hidden_size = new_hidden_size
        self.head_dim = new_head_dim
        
        # Reinitialize RoPE for new head dim
        self._init_rope()


class GrowableMLP(nn.Module):
    """MLP that can grow dimensions"""
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # SwiGLU style
        self.gate_proj = GrowableLinear(hidden_size, intermediate_size)
        self.up_proj = GrowableLinear(hidden_size, intermediate_size)
        self.down_proj = GrowableLinear(intermediate_size, hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
    
    def grow_hidden(self, new_hidden_size: int, new_intermediate_size: int):
        """Grow both hidden and intermediate dimensions"""
        self.gate_proj.grow_both(new_hidden_size, new_intermediate_size)
        self.up_proj.grow_both(new_hidden_size, new_intermediate_size)
        self.down_proj.grow_both(new_intermediate_size, new_hidden_size)
        
        self.hidden_size = new_hidden_size
        self.intermediate_size = new_intermediate_size


class GrowableExpert(nn.Module):
    """Single expert (MLP) that can grow"""
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.mlp = GrowableMLP(hidden_size, intermediate_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
    
    def grow_hidden(self, new_hidden_size: int, new_intermediate_size: int):
        self.mlp.grow_hidden(new_hidden_size, new_intermediate_size)


class GrowableMoE(nn.Module):
    """Mixture of Experts that can add experts dynamically"""
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 1,
        num_experts_per_token: int = 2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_token = min(num_experts_per_token, num_experts)
        
        # Router
        self.router = GrowableLinear(hidden_size, num_experts)
        
        # Experts
        self.experts = nn.ModuleList([
            GrowableExpert(hidden_size, intermediate_size)
            for _ in range(num_experts)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        
        if self.num_experts == 1:
            # Dense mode - just use the single expert
            return self.experts[0](x)
        
        # Flatten for routing
        x_flat = x.view(-1, hidden_size)
        
        # Route
        router_logits = self.router(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Top-k experts
        topk_probs, topk_indices = torch.topk(router_probs, self.num_experts_per_token, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        
        # Process through experts
        output = torch.zeros_like(x_flat)
        
        for i, expert in enumerate(self.experts):
            mask = (topk_indices == i).any(dim=-1)
            if mask.any():
                expert_input = x_flat[mask]
                expert_output = expert(expert_input)
                
                # Weight by router probability
                weights = topk_probs[mask]
                expert_weight = weights[topk_indices[mask] == i].unsqueeze(-1)
                output[mask] += expert_output * expert_weight.mean()
        
        return output.view(batch_size, seq_len, hidden_size)
    
    def add_expert(self):
        """Add a new expert"""
        new_expert = GrowableExpert(self.hidden_size, self.intermediate_size)
        
        # Initialize from average of existing experts
        if len(self.experts) > 0:
            with torch.no_grad():
                for new_param, existing_params in zip(
                    new_expert.parameters(),
                    zip(*[e.parameters() for e in self.experts])
                ):
                    avg_param = torch.stack(list(existing_params)).mean(dim=0)
                    noise = torch.randn_like(avg_param) * 0.01
                    new_param.copy_(avg_param + noise)
        
        self.experts.append(new_expert)
        self.num_experts += 1
        
        # Expand router
        self.router.grow_output(self.num_experts)
        
        # Update experts per token
        self.num_experts_per_token = min(self.num_experts_per_token, self.num_experts)
        
        logger.info(f"Added expert #{self.num_experts}")
    
    def grow_hidden(self, new_hidden_size: int, new_intermediate_size: int):
        """Grow all experts"""
        for expert in self.experts:
            expert.grow_hidden(new_hidden_size, new_intermediate_size)
        
        self.router.grow_input(new_hidden_size)
        
        self.hidden_size = new_hidden_size
        self.intermediate_size = new_intermediate_size


class GrowableTransformerBlock(nn.Module):
    """Transformer block that can grow"""
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        num_experts: int = 1,
    ):
        super().__init__()
        
        self.input_layernorm = GrowableRMSNorm(hidden_size)
        self.self_attn = GrowableAttention(hidden_size, num_heads, num_kv_heads)
        self.post_attention_layernorm = GrowableRMSNorm(hidden_size)
        self.mlp = GrowableMoE(hidden_size, intermediate_size, num_experts)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask, position_ids)
        hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states
    
    def grow_hidden(self, new_hidden_size: int, new_intermediate_size: int):
        """Grow hidden dimension"""
        self.input_layernorm.grow_hidden(new_hidden_size)
        self.self_attn.grow_hidden(new_hidden_size)
        self.post_attention_layernorm.grow_hidden(new_hidden_size)
        self.mlp.grow_hidden(new_hidden_size, new_intermediate_size)
    
    def add_expert(self):
        """Add an expert to the MoE layer"""
        self.mlp.add_expert()


class ContinuouslyGrowingAI(nn.Module):
    """
    An AI model that grows continuously.
    
    No discrete phases - just smooth, continuous expansion
    based on training progress and available compute.
    """
    
    def __init__(
        self,
        vocab_size: int = 32000,
        initial_hidden_size: int = 256,
        initial_layers: int = 4,
        initial_heads: int = 4,
        initial_kv_heads: int = 2,
        initial_experts: int = 1,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        
        # Growth state
        self.growth_state = GrowthState(
            hidden_size=initial_hidden_size,
            num_layers=initial_layers,
            num_heads=initial_heads,
            num_kv_heads=initial_kv_heads,
            intermediate_size=initial_hidden_size * 4,
            num_experts=initial_experts,
        )
        
        # Model components
        self.embed_tokens = GrowableEmbedding(vocab_size, initial_hidden_size)
        
        self.layers = nn.ModuleList([
            GrowableTransformerBlock(
                hidden_size=initial_hidden_size,
                intermediate_size=initial_hidden_size * 4,
                num_heads=initial_heads,
                num_kv_heads=initial_kv_heads,
                num_experts=initial_experts,
            )
            for _ in range(initial_layers)
        ])
        
        self.norm = GrowableRMSNorm(initial_hidden_size)
        self.lm_head = GrowableLinear(initial_hidden_size, vocab_size)
        
        # Track parameter count
        self._update_param_count()
        
        logger.info(f"ContinuouslyGrowingAI initialized: {self.param_count:,} params")
    
    def _update_param_count(self):
        """Update parameter count"""
        self.param_count = sum(p.numel() for p in self.parameters())
    
    def _create_causal_mask(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Create causal attention mask"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask.to(dtype)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        
        # Embed
        hidden_states = self.embed_tokens(input_ids)
        
        # Create causal mask
        causal_mask = self._create_causal_mask(seq_len, hidden_states.device, hidden_states.dtype)
        
        # Position IDs
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Process through layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, causal_mask, position_ids)
        
        # Final norm
        hidden_states = self.norm(hidden_states)
        
        # Get logits
        logits = self.lm_head(hidden_states)
        
        return {"logits": logits}
    
    def grow_hidden(self, increment: int = 64):
        """Grow hidden dimension by increment"""
        old_hidden = self.growth_state.hidden_size
        new_hidden = old_hidden + increment
        new_intermediate = new_hidden * 4
        
        # Grow all components
        self.embed_tokens.grow_hidden(new_hidden)
        
        for layer in self.layers:
            layer.grow_hidden(new_hidden, new_intermediate)
        
        self.norm.grow_hidden(new_hidden)
        self.lm_head.grow_input(new_hidden)
        
        # Update state
        self.growth_state.hidden_size = new_hidden
        self.growth_state.intermediate_size = new_intermediate
        self.growth_state.growth_events.append({
            "type": "hidden_growth",
            "old_size": old_hidden,
            "new_size": new_hidden,
            "tokens_trained": self.growth_state.total_tokens_trained,
            "timestamp": time.time(),
        })
        
        self._update_param_count()
        logger.info(f"Hidden dimension grown: {old_hidden} → {new_hidden} ({self.param_count:,} params)")
    
    def add_layer(self):
        """Add a new transformer layer"""
        # Create new layer with current dimensions
        new_layer = GrowableTransformerBlock(
            hidden_size=self.growth_state.hidden_size,
            intermediate_size=self.growth_state.intermediate_size,
            num_heads=self.growth_state.num_heads,
            num_kv_heads=self.growth_state.num_kv_heads,
            num_experts=self.growth_state.num_experts,
        )
        
        # Initialize from interpolation of existing layers
        if len(self.layers) >= 2:
            with torch.no_grad():
                # Average the middle layers
                mid = len(self.layers) // 2
                for new_param, p1, p2 in zip(
                    new_layer.parameters(),
                    self.layers[mid-1].parameters(),
                    self.layers[mid].parameters(),
                ):
                    avg = (p1 + p2) / 2
                    noise = torch.randn_like(avg) * 0.01
                    new_param.copy_(avg + noise)
        
        # Insert in middle
        mid = len(self.layers) // 2
        self.layers.insert(mid, new_layer)
        
        old_layers = self.growth_state.num_layers
        self.growth_state.num_layers += 1
        self.growth_state.growth_events.append({
            "type": "layer_added",
            "old_count": old_layers,
            "new_count": self.growth_state.num_layers,
            "tokens_trained": self.growth_state.total_tokens_trained,
            "timestamp": time.time(),
        })
        
        self._update_param_count()
        logger.info(f"Layer added: {old_layers} → {self.growth_state.num_layers} ({self.param_count:,} params)")
    
    def add_expert(self):
        """Add an expert to all MoE layers"""
        for layer in self.layers:
            layer.add_expert()
        
        old_experts = self.growth_state.num_experts
        self.growth_state.num_experts += 1
        self.growth_state.growth_events.append({
            "type": "expert_added",
            "old_count": old_experts,
            "new_count": self.growth_state.num_experts,
            "tokens_trained": self.growth_state.total_tokens_trained,
            "timestamp": time.time(),
        })
        
        self._update_param_count()
        logger.info(f"Expert added to all layers: {old_experts} → {self.growth_state.num_experts} ({self.param_count:,} params)")
    
    def maybe_grow(self, tokens_this_step: int) -> List[str]:
        """
        Check if model should grow and apply growth.
        Call this after each training step.
        
        Returns list of growth events that occurred.
        """
        self.growth_state.total_tokens_trained += tokens_this_step
        events = []
        
        # Check each growth type
        if self.growth_state.should_grow_hidden():
            self.grow_hidden()
            events.append("hidden")
        
        if self.growth_state.should_grow_layer():
            self.add_layer()
            events.append("layer")
        
        if self.growth_state.should_grow_expert():
            self.add_expert()
            events.append("expert")
        
        return events
    
    def set_compute_limits(self, gpu_memory_gb: float):
        """
        Set maximum model size based on available GPU memory.
        
        Rough estimates:
        - 4GB: ~500M params
        - 8GB: ~1B params
        - 16GB: ~3B params
        - 24GB: ~7B params
        - 48GB: ~20B params
        - 80GB: ~40B params
        """
        # Estimate max params (conservative: ~4 bytes/param * 4 for optimizer states)
        max_params = int(gpu_memory_gb * 1e9 / 16)  # 16 bytes per param with optimizer
        
        # Calculate max dimensions
        # Rough formula: params ≈ 12 * L * H^2 (for transformer)
        # Solve for H given L and target params
        
        if max_params < 100_000_000:
            self.growth_state.max_hidden_size = 768
            self.growth_state.max_layers = 12
            self.growth_state.max_experts = 1
        elif max_params < 1_000_000_000:
            self.growth_state.max_hidden_size = 2048
            self.growth_state.max_layers = 24
            self.growth_state.max_experts = 4
        elif max_params < 10_000_000_000:
            self.growth_state.max_hidden_size = 4096
            self.growth_state.max_layers = 32
            self.growth_state.max_experts = 8
        elif max_params < 100_000_000_000:
            self.growth_state.max_hidden_size = 8192
            self.growth_state.max_layers = 80
            self.growth_state.max_experts = 16
        else:
            self.growth_state.max_hidden_size = 16384
            self.growth_state.max_layers = 128
            self.growth_state.max_experts = 64
        
        logger.info(
            f"Compute limits set: max_hidden={self.growth_state.max_hidden_size}, "
            f"max_layers={self.growth_state.max_layers}, max_experts={self.growth_state.max_experts}"
        )
    
    def get_growth_status(self) -> Dict[str, Any]:
        """Get current growth status"""
        return {
            "current_params": f"{self.param_count:,}",
            "hidden_size": self.growth_state.hidden_size,
            "num_layers": self.growth_state.num_layers,
            "num_heads": self.growth_state.num_heads,
            "num_experts": self.growth_state.num_experts,
            "tokens_trained": f"{self.growth_state.total_tokens_trained:,}",
            "growth_events": len(self.growth_state.growth_events),
            "limits": {
                "max_hidden": self.growth_state.max_hidden_size,
                "max_layers": self.growth_state.max_layers,
                "max_experts": self.growth_state.max_experts,
            },
            "next_growth": {
                "hidden_at": f"{(self.growth_state.hidden_size // self.growth_state.hidden_growth_increment + 1) * self.growth_state.tokens_per_hidden_growth:,} tokens",
                "layer_at": f"{(self.growth_state.num_layers + 1) * self.growth_state.tokens_per_layer_growth:,} tokens",
                "expert_at": f"{(self.growth_state.num_experts + 1) * self.growth_state.tokens_per_expert_growth:,} tokens" if self.growth_state.hidden_size >= 1024 else "Need hidden>=1024",
            },
        }


class ContinuousGrowthTrainer:
    """
    Trainer that manages continuous model growth.
    
    The model grows automatically as it trains -
    no manual intervention needed.
    """
    
    def __init__(
        self,
        model: ContinuouslyGrowingAI,
        learning_rate: float = 3e-4,
    ):
        self.model = model
        self.base_lr = learning_rate
        self.optimizer = None
        self.scheduler = None
        
        self._init_optimizer()
    
    def _init_optimizer(self):
        """Initialize or reinitialize optimizer after growth"""
        # Scale learning rate with model size
        # Larger models need smaller learning rates
        param_count = self.model.param_count
        
        if param_count < 100_000_000:
            lr = self.base_lr
        elif param_count < 1_000_000_000:
            lr = self.base_lr * 0.5
        elif param_count < 10_000_000_000:
            lr = self.base_lr * 0.1
        else:
            lr = self.base_lr * 0.03
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )
        
        logger.info(f"Optimizer initialized with lr={lr}")
    
    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Single training step with automatic growth check.
        """
        self.model.train()
        
        # Forward pass
        outputs = self.model(input_ids)
        logits = outputs["logits"]
        
        # Compute loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Check for growth
        tokens_this_step = input_ids.numel()
        growth_events = self.model.maybe_grow(tokens_this_step)
        
        # Reinitialize optimizer if model grew
        if growth_events:
            self._init_optimizer()
        
        return {
            "loss": loss.item(),
            "tokens": tokens_this_step,
            "growth_events": growth_events,
            "params": self.model.param_count,
        }


if __name__ == "__main__":
    # Demo continuous growth
    logging.basicConfig(level=logging.INFO)
    
    print("=== Continuous Growth Demo ===\n")
    
    # Create model
    model = ContinuouslyGrowingAI(
        vocab_size=32000,
        initial_hidden_size=256,
        initial_layers=4,
    )
    
    # Set limits based on GPU
    model.set_compute_limits(gpu_memory_gb=4.0)  # 4GB GPU
    
    print("\nInitial state:")
    status = model.get_growth_status()
    for k, v in status.items():
        print(f"  {k}: {v}")
    
    # Simulate training
    print("\n--- Simulating training ---")
    
    # Simulate 100M tokens of training
    tokens_per_step = 2048 * 32  # batch_size * seq_len
    
    for step in range(1500):
        events = model.maybe_grow(tokens_per_step)
        
        if events:
            print(f"\nStep {step}: Growth events: {events}")
            status = model.get_growth_status()
            print(f"  Params: {status['current_params']}")
            print(f"  Hidden: {status['hidden_size']}")
            print(f"  Layers: {status['num_layers']}")
    
    print("\n--- Final State ---")
    status = model.get_growth_status()
    for k, v in status.items():
        print(f"  {k}: {v}")
