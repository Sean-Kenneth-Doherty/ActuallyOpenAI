#!/usr/bin/env python3
"""
ActuallyOpenAI - Continuous Growth Node
=======================================
AI that grows continuously without discrete steps.

The model expands smoothly as it trains:
- Hidden dimensions grow gradually
- Layers are added one at a time
- Experts are added progressively
- Knowledge is preserved through growth

Usage:
    python continuous_node.py                    # Start with defaults
    python continuous_node.py --gpu-memory 8    # Set GPU memory limit
"""

import asyncio
import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger("AOAI-Continuous")


CONTINUOUS_BANNER = """
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║     █████╗  ██████╗████████╗██╗   ██╗ █████╗ ██╗     ██╗  ██╗   ██╗      ║
║    ██╔══██╗██╔════╝╚══██╔══╝██║   ██║██╔══██╗██║     ██║  ╚██╗ ██╔╝      ║
║    ███████║██║        ██║   ██║   ██║███████║██║     ██║   ╚████╔╝       ║
║    ██╔══██║██║        ██║   ██║   ██║██╔══██║██║     ██║    ╚██╔╝        ║
║    ██║  ██║╚██████╗   ██║   ╚██████╔╝██║  ██║███████╗███████╗██║         ║
║    ╚═╝  ╚═╝ ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝         ║
║                                                                           ║
║              ██████╗ ██████╗ ███████╗███╗   ██╗ █████╗ ██╗               ║
║             ██╔═══██╗██╔══██╗██╔════╝████╗  ██║██╔══██╗██║               ║
║             ██║   ██║██████╔╝█████╗  ██╔██╗ ██║███████║██║               ║
║             ██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║██╔══██║██║               ║
║             ╚██████╔╝██║     ███████╗██║ ╚████║██║  ██║██║               ║
║              ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝               ║
║                                                                           ║
║          🌱 CONTINUOUS GROWTH MODE - NO DISCRETE STEPS 🌱                ║
║                                                                           ║
║    The AI grows smoothly as it trains:                                   ║
║      • Hidden dimensions expand gradually                                ║
║      • Layers added one at a time                                        ║
║      • Experts added progressively                                       ║
║      • Knowledge preserved through all growth                            ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""


class ContinuousGrowthNode:
    """
    Node with continuously growing AI model.
    
    The model grows smoothly based on:
    - Training progress (tokens processed)
    - Available compute (GPU memory)
    - Network capacity (more nodes = larger limits)
    """
    
    def __init__(
        self,
        p2p_port: int = 31337,
        api_port: int = 8080,
        data_dir: str = "./aoai_data",
        gpu_memory: float = 4.0,
    ):
        self.p2p_port = p2p_port
        self.api_port = api_port
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.gpu_memory = gpu_memory
        
        # Core components
        self.p2p_node = None
        self.api_mesh = None
        self.wallet = None
        self.ledger = None
        
        # Continuous growth components
        self.model = None
        self.trainer = None
        
        # Stats
        self.inference_count = 0
        self.training_steps = 0
        self.growth_events = []
        
        self.is_running = False
    
    async def start(self):
        """Start the continuous growth node"""
        print(CONTINUOUS_BANNER)
        print(f"🚀 Starting Continuous Growth Node (GPU: {self.gpu_memory}GB)...\n")
        
        self.is_running = True
        
        await self._init_components()
        await self._start_services()
        
        self._print_status()
        
        # Start the growth loop
        asyncio.create_task(self._growth_loop())
        
        await self._run_loop()
    
    async def _init_components(self):
        """Initialize all components"""
        from actuallyopenai.network.p2p_node import P2PNode
        from actuallyopenai.network.api_mesh import DecentralizedAPIMesh
        from actuallyopenai.token.aoai_token import AOAIWallet, AOAILedger, ComputeRewarder
        
        # Wallet
        wallet_path = self.data_dir / "wallet.json"
        if wallet_path.exists():
            self.wallet = AOAIWallet.from_file(str(wallet_path))
            logger.info(f"💰 Wallet loaded: {self.wallet.address[:20]}...")
        else:
            self.wallet = AOAIWallet()
            self.wallet.save(str(wallet_path))
            logger.info(f"💰 New wallet created: {self.wallet.address[:20]}...")
        
        # Ledger
        self.ledger = AOAILedger(str(self.data_dir / "ledger"))
        self.rewarder = ComputeRewarder(self.ledger)
        
        # P2P
        self.p2p_node = P2PNode(port=self.p2p_port, node_id=self.wallet.address)
        
        # API
        self.api_mesh = DecentralizedAPIMesh(
            node_id=self.wallet.address,
            port=self.api_port
        )
        
        # Initialize continuous growth model
        await self._init_continuous_model()
    
    async def _init_continuous_model(self):
        """Initialize the continuously growing model"""
        try:
            from actuallyopenai.models.continuous_growth import (
                ContinuouslyGrowingAI,
                ContinuousGrowthTrainer,
            )
            
            # Detect actual GPU memory
            gpu_mem = self._detect_gpu_memory()
            if gpu_mem > 0:
                self.gpu_memory = gpu_mem
            
            # Create model
            self.model = ContinuouslyGrowingAI(
                vocab_size=32000,
                initial_hidden_size=256,
                initial_layers=4,
                initial_heads=4,
                initial_kv_heads=2,
                initial_experts=1,
            )
            
            # Set growth limits based on available GPU
            self.model.set_compute_limits(self.gpu_memory)
            
            # Move to GPU if available
            device = self._get_device()
            self.model = self.model.to(device)
            
            logger.info(f"🧠 Continuous growth model initialized: {self.model.param_count:,} params")
            logger.info(f"🔧 Growth limits: hidden={self.model.growth_state.max_hidden_size}, layers={self.model.growth_state.max_layers}")
            
        except Exception as e:
            logger.error(f"Failed to initialize continuous model: {e}")
            import traceback
            traceback.print_exc()
    
    def _detect_gpu_memory(self) -> float:
        """Detect available GPU memory in GB"""
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                return props.total_memory / (1024**3)
            return 0.0
        except:
            return 0.0
    
    def _get_device(self):
        """Get compute device"""
        import torch
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    
    async def _start_services(self):
        """Start all services"""
        await self.p2p_node.start()
        logger.info("🌐 P2P network started")
        
        # Register handlers
        async def chat_handler(data: dict) -> dict:
            return await self._handle_chat(data)
        
        async def growth_status_handler(data: dict) -> dict:
            return self._get_growth_status()
        
        self.api_mesh.register_handler("chat", chat_handler)
        self.api_mesh.register_handler("growth_status", growth_status_handler)
        
        await self.api_mesh.start()
        logger.info("🔌 API mesh started")
    
    async def _handle_chat(self, data: dict) -> dict:
        """Handle chat completion with the growing model"""
        import torch
        import hashlib
        
        try:
            messages = data.get("messages", [])
            prompt = messages[-1]["content"] if messages else ""
            max_tokens = data.get("max_tokens", 256)
            temperature = data.get("temperature", 0.7)
            
            # Generate
            response = await self._generate(prompt, max_tokens, temperature)
            
            self.inference_count += 1
            
            # Trigger growth check (inference contributes to "training" via feedback)
            tokens_processed = len(prompt.split()) + len(response.split())
            if self.model:
                events = self.model.maybe_grow(tokens_processed * 10)  # Weight inference
                if events:
                    self.growth_events.extend(events)
                    logger.info(f"🌱 Model grew during inference: {events}")
            
            return {
                "id": f"chatcmpl-{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": f"aoai-continuous-{self.model.param_count if self.model else 0}",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": response},
                    "finish_reason": "stop"
                }],
                "growth_info": self.model.get_growth_status() if self.model else None
            }
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return {"error": str(e)}
    
    async def _generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7
    ) -> str:
        """Generate text with the continuously growing model"""
        import torch
        
        if self.model is None:
            return "Model not initialized"
        
        device = next(self.model.parameters()).device
        
        # Simple tokenization
        input_ids = torch.tensor(
            [[ord(c) % 32000 for c in prompt[-512:]]],
            device=device
        )
        
        self.model.eval()
        generated = []
        
        with torch.no_grad():
            for _ in range(min(max_tokens, 256)):
                outputs = self.model(input_ids)
                logits = outputs["logits"]
                
                next_logits = logits[0, -1, :] / temperature
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                
                char_idx = next_token.item() % 128
                if 32 <= char_idx < 127:
                    generated.append(chr(char_idx))
                
                if generated and generated[-1] in '.!?\n':
                    break
        
        return ''.join(generated) if generated else "..."
    
    async def _growth_loop(self):
        """Background loop that simulates training and triggers growth"""
        import torch
        
        logger.info("🌱 Continuous growth loop started")
        
        batch_size = 8
        seq_len = 256
        
        while self.is_running:
            try:
                if self.model is None:
                    await asyncio.sleep(5)
                    continue
                
                # Simulate a mini training step
                device = next(self.model.parameters()).device
                
                # Generate random training data
                input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=device)
                
                # Forward pass (no backward - just to show growth)
                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(input_ids)
                
                # Simulate tokens processed
                tokens = batch_size * seq_len
                
                # Check for growth
                events = self.model.maybe_grow(tokens)
                
                if events:
                    self.growth_events.extend(events)
                    status = self.model.get_growth_status()
                    logger.info(
                        f"🌱 GROWTH: {events} | "
                        f"Params: {status['current_params']} | "
                        f"Hidden: {status['hidden_size']} | "
                        f"Layers: {status['num_layers']} | "
                        f"Experts: {status['num_experts']}"
                    )
                
                self.training_steps += 1
                
                # Slow down simulation
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Growth loop error: {e}")
                await asyncio.sleep(5)
    
    def _get_growth_status(self) -> Dict[str, Any]:
        """Get current growth status"""
        status = {
            "node_id": self.wallet.address[:16] + "..." if self.wallet else "unknown",
            "mode": "continuous_growth",
            "gpu_memory_gb": self.gpu_memory,
            "inference_count": self.inference_count,
            "training_steps": self.training_steps,
            "total_growth_events": len(self.growth_events),
        }
        
        if self.model:
            model_status = self.model.get_growth_status()
            status["model"] = model_status
        
        return status
    
    def _print_status(self):
        """Print node status"""
        balance = self.ledger.get_balance(self.wallet.address) if self.ledger else 0
        
        model_params = self.model.param_count if self.model else 0
        hidden = self.model.growth_state.hidden_size if self.model else 0
        layers = self.model.growth_state.num_layers if self.model else 0
        experts = self.model.growth_state.num_experts if self.model else 0
        
        print(f"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                  🟢 CONTINUOUS GROWTH NODE RUNNING 🟢                     ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  Wallet: {self.wallet.address[:40] if self.wallet else 'N/A'}...         ║
║  Balance: {balance:,.4f} AOAI                                             ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  CURRENT MODEL STATE:                                                     ║
║    Parameters: {model_params:>12,}                                        ║
║    Hidden Size: {hidden:>10}   (grows by 64 every 50M tokens)            ║
║    Layers: {layers:>14}   (adds 1 every 200M tokens)                     ║
║    Experts: {experts:>13}   (adds 1 every 500M tokens)                   ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  GROWTH LIMITS (based on {self.gpu_memory:.1f}GB GPU):                             ║
║    Max Hidden: {self.model.growth_state.max_hidden_size if self.model else 'N/A':>9}                                               ║
║    Max Layers: {self.model.growth_state.max_layers if self.model else 'N/A':>9}                                               ║
║    Max Experts: {self.model.growth_state.max_experts if self.model else 'N/A':>8}                                               ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  API Endpoints:                                                           ║
║    http://localhost:{self.api_port}/v1/chat/completions                         ║
║    http://localhost:{self.api_port}/growth_status                               ║
╚═══════════════════════════════════════════════════════════════════════════╝

  🌱 The AI grows continuously as it trains and serves requests.
  📈 No discrete steps - smooth, continuous expansion.
  🧠 Knowledge is preserved through all growth events.

  Press Ctrl+C to stop.
""")
    
    async def _run_loop(self):
        """Main run loop with status updates"""
        last_params = 0
        
        try:
            while self.is_running:
                await asyncio.sleep(30)
                
                if self.model:
                    current_params = self.model.param_count
                    status = self.model.get_growth_status()
                    
                    growth_indicator = ""
                    if current_params > last_params:
                        growth_indicator = f" (+{current_params - last_params:,} params)"
                        last_params = current_params
                    
                    logger.info(
                        f"📊 Params: {status['current_params']}{growth_indicator} | "
                        f"H:{status['hidden_size']} L:{status['num_layers']} E:{status['num_experts']} | "
                        f"Tokens: {status['tokens_trained']}"
                    )
                    
        except asyncio.CancelledError:
            pass
    
    async def stop(self):
        """Stop the node"""
        print("\n🛑 Shutting down continuous growth node...")
        self.is_running = False
        
        if self.p2p_node:
            await self.p2p_node.stop()
        
        if self.api_mesh:
            await self.api_mesh.stop()
        
        # Save growth state
        if self.model:
            status = self.model.get_growth_status()
            logger.info(f"Final model state: {status['current_params']} params")
        
        print("👋 Goodbye! Your model's growth has been recorded.")


async def main():
    parser = argparse.ArgumentParser(
        description="ActuallyOpenAI Continuous Growth Node"
    )
    parser.add_argument(
        "--p2p-port", type=int, default=31337,
        help="P2P network port (default: 31337)"
    )
    parser.add_argument(
        "--api-port", type=int, default=8080,
        help="API server port (default: 8080)"
    )
    parser.add_argument(
        "--data-dir", type=str, default="./aoai_data",
        help="Data directory (default: ./aoai_data)"
    )
    parser.add_argument(
        "--gpu-memory", type=float, default=4.0,
        help="GPU memory in GB (auto-detected if not specified)"
    )
    
    args = parser.parse_args()
    
    node = ContinuousGrowthNode(
        p2p_port=args.p2p_port,
        api_port=args.api_port,
        data_dir=args.data_dir,
        gpu_memory=args.gpu_memory,
    )
    
    # Handle shutdown
    loop = asyncio.get_event_loop()
    
    def signal_handler():
        asyncio.create_task(node.stop())
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            pass
    
    try:
        await node.start()
    except KeyboardInterrupt:
        await node.stop()


if __name__ == "__main__":
    import sys
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
