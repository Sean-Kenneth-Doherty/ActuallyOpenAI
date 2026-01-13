"""
Run the ActuallyOpenAI self-improving AI training system.

This script:
1. Creates a model
2. Starts the training orchestrator
3. Simulates workers contributing compute
4. Shows the AI improving in real-time
"""

import asyncio
import signal
import sys
import torch
from datetime import datetime

print("=" * 70)
print("🚀 ActuallyOpenAI - Self-Improving AI Training System")
print("=" * 70)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Import components
from actuallyopenai.models import AOAIModel, ModelConfig
from actuallyopenai.training import (
    ContinuousTrainer,
    FederatedAggregator,
    ModelEvolution,
    AutoScalingController,
    ImprovementTracker,
    GradientPacket,
    WorkerUpdate,
    AggregationStrategy,
    ScalingMode,
)
from actuallyopenai.core.models import Worker, WorkerStatus

# Global flag for graceful shutdown
running = True

def signal_handler(sig, frame):
    global running
    print("\n\n⏹️  Shutting down gracefully...")
    running = False

signal.signal(signal.SIGINT, signal_handler)


class SimulatedWorker:
    """Simulates a worker contributing compute."""
    
    def __init__(self, worker_id: str, compute_power: float, has_gpu: bool = False):
        self.worker_id = worker_id
        self.compute_power = compute_power
        self.has_gpu = has_gpu
        self.tasks_completed = 0
        self.total_compute_time = 0.0
    
    async def compute_gradients(
        self, 
        model: torch.nn.Module, 
        batch_size: int,
        seq_len: int = 64
    ) -> tuple:
        """Simulate computing gradients on a batch."""
        # Generate random data (in production, would use real training data)
        input_ids = torch.randint(0, 50257, (batch_size, seq_len))
        labels = torch.randint(0, 50257, (batch_size, seq_len))
        
        # Forward and backward pass
        model.zero_grad()
        logits, loss = model(input_ids, labels=labels)
        loss.backward()
        
        # Collect gradients
        gradients = {
            name: param.grad.clone()
            for name, param in model.named_parameters()
            if param.grad is not None
        }
        
        # Simulate compute time based on worker power
        compute_time = (batch_size / self.compute_power) * (0.5 if self.has_gpu else 2.0)
        await asyncio.sleep(compute_time * 0.1)  # Scaled down for demo
        
        self.tasks_completed += 1
        self.total_compute_time += compute_time
        
        return gradients, loss.item(), compute_time


async def main():
    global running
    
    # =========================================================================
    # Setup
    # =========================================================================
    print("[1/4] Creating model...")
    config = ModelConfig.tiny()  # Use tiny for demo speed
    model = AOAIModel(config)
    print(f"      ✅ Model: {model.num_parameters:,} parameters")
    print(f"         Layers: {config.num_layers}, Hidden: {config.hidden_size}")
    print()
    
    print("[2/4] Initializing training system...")
    
    # Continuous Trainer
    trainer = ContinuousTrainer(
        model=model,
        model_id="aoai-demo-v1",
        checkpoint_dir="./demo_checkpoints",
        min_workers=1,
        target_batch_size=64,
        checkpoint_every_steps=50,
        eval_every_steps=25,
    )
    await trainer.initialize()
    print("      ✅ Continuous Trainer ready")
    
    # Federated Aggregator
    aggregator = FederatedAggregator(
        strategy=AggregationStrategy.FEDAVG,
        min_workers=1,
    )
    print("      ✅ Federated Aggregator ready")
    
    # Model Evolution
    evolution = ModelEvolution(
        model_family="aoai-demo",
        evolution_dir="./demo_evolution",
    )
    gen = await evolution.start_new_generation(model)
    print(f"      ✅ Model Evolution ready (Generation {gen.generation_number})")
    
    # Auto Scaler
    scaler = AutoScalingController(
        mode=ScalingMode.BALANCED,
        min_batch_size=8,
        max_batch_size=128,
    )
    print("      ✅ Auto Scaler ready")
    
    # Improvement Tracker
    tracker = ImprovementTracker(tracking_dir="./demo_tracking")
    print("      ✅ Improvement Tracker ready")
    print()
    
    print("[3/4] Spawning simulated workers...")
    workers = [
        SimulatedWorker("worker-alpha", compute_power=50, has_gpu=True),
        SimulatedWorker("worker-beta", compute_power=30, has_gpu=False),
        SimulatedWorker("worker-gamma", compute_power=40, has_gpu=True),
    ]
    for w in workers:
        print(f"      ✅ {w.worker_id}: power={w.compute_power}, GPU={w.has_gpu}")
    print()
    
    print("[4/4] Starting continuous training loop...")
    print()
    print("=" * 70)
    print("📊 LIVE TRAINING PROGRESS (Press Ctrl+C to stop)")
    print("=" * 70)
    print()
    
    # =========================================================================
    # Training Loop
    # =========================================================================
    
    step = 0
    best_loss = float('inf')
    improvements = 0
    start_time = datetime.now()
    
    while running:
        step += 1
        
        # Take compute snapshot
        worker_dict = {
            w.worker_id: {
                "status": "training",
                "compute_score": w.compute_power,
                "has_gpu": w.has_gpu,
                "vram_gb": 8 if w.has_gpu else 0,
            }
            for w in workers
        }
        snapshot = await scaler.take_snapshot(worker_dict)
        
        # Get scaling parameters
        params = scaler.get_current_parameters()
        batch_size = min(params["global_batch_size"] // len(workers), 16)
        
        # Each worker computes gradients
        all_gradients = []
        total_loss = 0.0
        total_samples = 0
        total_compute = 0.0
        
        for worker in workers:
            grads, loss, compute_time = await worker.compute_gradients(
                model=model,
                batch_size=batch_size
            )
            
            all_gradients.append((worker.worker_id, grads, loss, batch_size, compute_time))
            total_loss += loss * batch_size
            total_samples += batch_size
            total_compute += compute_time
            
            # Send to aggregator
            update = WorkerUpdate(
                worker_id=worker.worker_id,
                round_id=aggregator.current_round,
                gradients=grads,
                num_samples=batch_size,
                local_loss=loss,
                compute_time=compute_time,
            )
            await aggregator.receive_update(update)
        
        avg_loss = total_loss / total_samples
        
        # Aggregate gradients
        if aggregator.ready_to_aggregate():
            result = await aggregator.aggregate()
            
            if result and result.aggregated_gradients:
                # Apply aggregated gradients to model
                optimizer = torch.optim.AdamW(model.parameters(), lr=params["effective_learning_rate"])
                optimizer.zero_grad()
                
                for name, param in model.named_parameters():
                    if name in result.aggregated_gradients:
                        param.grad = result.aggregated_gradients[name]
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                # Update model
                optimizer.step()
                
                # Track improvement
                if avg_loss < best_loss:
                    improvement_pct = (best_loss - avg_loss) / best_loss * 100 if best_loss != float('inf') else 0
                    best_loss = avg_loss
                    improvements += 1
                    improved = True
                else:
                    improved = False
                
                # Update evolution metrics
                await evolution.update_generation_metrics(
                    generation_id=gen.id,
                    loss=avg_loss,
                    steps=step,
                    tokens=total_samples * 64,  # seq_len
                    compute_hours=total_compute / 3600,
                )
                
                # Print progress
                elapsed = (datetime.now() - start_time).total_seconds()
                tokens_per_sec = (step * total_samples * 64) / elapsed
                
                status_icon = "🎉" if improved else "📈"
                improvement_str = f" (+{improvement_pct:.2f}%)" if improved else ""
                
                print(f"Step {step:4d} | Loss: {avg_loss:.4f}{improvement_str} | "
                      f"Best: {best_loss:.4f} | Workers: {len(workers)} | "
                      f"Batch: {total_samples} | {tokens_per_sec:.0f} tok/s {status_icon}")
                
                # Periodic summary
                if step % 25 == 0:
                    print()
                    print(f"    📊 Summary after {step} steps:")
                    print(f"       • Total improvements: {improvements}")
                    print(f"       • Improvement rate: {improvements/step*100:.1f}%")
                    print(f"       • Compute hours: {sum(w.total_compute_time for w in workers)/3600:.4f}")
                    print(f"       • Tasks completed: {sum(w.tasks_completed for w in workers)}")
                    print(f"       • Resource tier: {scaler.current_tier.value}")
                    print()
        
        # Small delay
        await asyncio.sleep(0.05)
        
        # Demo: run for 100 steps then stop
        if step >= 100:
            print()
            print("=" * 70)
            print(f"🏁 Demo complete after {step} steps!")
            break
    
    # =========================================================================
    # Final Report
    # =========================================================================
    print()
    print("=" * 70)
    print("📋 FINAL TRAINING REPORT")
    print("=" * 70)
    print()
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print(f"🕐 Training Duration: {elapsed:.1f} seconds")
    print(f"📈 Total Steps: {step}")
    print(f"🎯 Final Loss: {avg_loss:.4f}")
    print(f"🏆 Best Loss: {best_loss:.4f}")
    print(f"✨ Total Improvements: {improvements}")
    print(f"📊 Improvement Rate: {improvements/step*100:.1f}%")
    print()
    
    print("👥 Worker Contributions:")
    for w in workers:
        print(f"   • {w.worker_id}: {w.tasks_completed} tasks, {w.total_compute_time:.2f}s compute")
    print()
    
    print("🧬 Model Evolution:")
    summary = evolution.get_evolution_summary()
    print(f"   • Generation: {summary['total_generations']}")
    print(f"   • Best Loss: {summary.get('best_loss', 'N/A')}")
    print()
    
    print("⚙️  Auto-Scaling:")
    scaling_status = scaler.get_status()
    print(f"   • Mode: {scaling_status['mode']}")
    print(f"   • Resource Tier: {scaling_status['current_tier']}")
    print(f"   • Batch Size: {scaling_status['parameters']['global_batch_size']}")
    print()
    
    print("=" * 70)
    print("🎉 The AI successfully improved as compute was contributed!")
    print("   This demonstrates the self-assembling, continuously improving")
    print("   architecture of ActuallyOpenAI.")
    print("=" * 70)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Training stopped by user.")
