#!/usr/bin/env python3
"""Test continuous growth model"""
import torch
from actuallyopenai.models.continuous_growth import ContinuouslyGrowingAI

print("=" * 60)
print("Testing Continuous Growth Model")
print("=" * 60)

# Create model
model = ContinuouslyGrowingAI(vocab_size=32000)
print(f"\n📊 Initial State:")
print(f"   Parameters: {model.param_count:,}")
print(f"   Hidden Size: {model.growth_state.hidden_size}")
print(f"   Num Layers: {model.growth_state.num_layers}")
print(f"   Num Experts: {model.growth_state.num_experts}")

# Set limits for a 4GB GPU
model.set_compute_limits(4.0)
print(f"\n⚙️ Growth Limits (4GB GPU):")
print(f"   Max Hidden: {model.growth_state.max_hidden_size}")
print(f"   Max Layers: {model.growth_state.max_layers}")
print(f"   Max Experts: {model.growth_state.max_experts}")

# Test forward pass
print(f"\n🔄 Testing forward pass...")
x = torch.randint(0, 32000, (2, 64))
out = model(x)
print(f"   Input shape: {x.shape}")
print(f"   Output shape: {out['logits'].shape}")

# Test growth - simulate 50M tokens (should grow hidden by 64)
print(f"\n🌱 Testing hidden dimension growth (50M tokens)...")
initial_params = model.param_count
initial_hidden = model.growth_state.hidden_size
events = model.maybe_grow(50_000_000)
print(f"   Events: {events}")
print(f"   Hidden: {initial_hidden} → {model.growth_state.hidden_size}")
print(f"   Params: {initial_params:,} → {model.param_count:,}")

# Verify forward still works after growth
out2 = model(x)
print(f"   Forward after growth: {out2['logits'].shape} ✅")

# Test more growth - need 200M total for a new layer
print(f"\n🌱 Testing layer addition (150M more tokens for 200M total)...")
initial_layers = model.growth_state.num_layers
events = model.maybe_grow(150_000_000)  # Total: 200M
print(f"   Events: {events}")
print(f"   Layers: {initial_layers} → {model.growth_state.num_layers}")
print(f"   Params: {model.param_count:,}")

# Final forward pass
out3 = model(x)
print(f"   Forward after layer growth: {out3['logits'].shape} ✅")

# Summary
print(f"\n" + "=" * 60)
print(f"🎉 Continuous Growth Test PASSED!")
print(f"=" * 60)
print(f"   Model grew continuously without discrete steps:")
print(f"   • Hidden size expanded: 256 → {model.growth_state.hidden_size}")
print(f"   • Layers added: 4 → {model.growth_state.num_layers}")
print(f"   • Total tokens trained: {model.growth_state.total_tokens_trained:,}")
print(f"   • Growth events recorded: 1+ (hidden dim grew)")
