#!/usr/bin/env python3
"""
ActuallyOpenAI - Scaling Node
=============================
Extended node with progressive scaling to frontier models.

This node can:
- Run inference on current model
- Contribute to distributed training
- Participate in network compute aggregation
- Scale to larger models as network grows

Usage:
    python scaling_node.py                    # Start with defaults
    python scaling_node.py --scale tiny       # Start at tiny scale
    python scaling_node.py --scale small      # Start at small scale (100M params)
"""

import asyncio
import argparse
import logging
import signal
import sys
from pathlib import Path
from typing import Optional

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger("AOAI-Scaling")


SCALING_BANNER = """
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
║           🚀 FRONTIER SCALING MODE - UNCAPPED INTELLIGENCE 🚀            ║
║                                                                           ║
║         tiny(10M) → small(100M) → medium(1B) → large(7B) → ...           ║
║                    → xlarge(70B) → FRONTIER(400B+)                       ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""


class ScalingNode:
    """
    Extended node with progressive scaling capabilities.
    
    This is the foundation for scaling to frontier-level models
    through distributed training across the network.
    """
    
    def __init__(
        self,
        p2p_port: int = 31337,
        api_port: int = 8080,
        data_dir: str = "./aoai_data",
        initial_scale: str = "tiny",
        enable_scaling: bool = True,
    ):
        self.p2p_port = p2p_port
        self.api_port = api_port
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.initial_scale = initial_scale
        self.enable_scaling = enable_scaling
        
        # Core components
        self.p2p_node = None
        self.api_mesh = None
        self.wallet = None
        self.ledger = None
        
        # Scaling components
        self.scalable_model = None
        self.scaling_orchestrator = None
        self.compute_aggregator = None
        self.improvement_engine = None
        
        self.is_running = False
    
    async def start(self):
        """Start the scaling node"""
        print(SCALING_BANNER)
        print(f"🚀 Starting Scaling Node (initial: {self.initial_scale.upper()})...\n")
        
        self.is_running = True
        
        # Initialize all components
        await self._init_components()
        
        # Start services
        await self._start_services()
        
        # Print status
        self._print_status()
        
        # Run main loop
        await self._run_loop()
    
    async def _init_components(self):
        """Initialize all components including scaling infrastructure"""
        from actuallyopenai.network.p2p_node import P2PNode
        from actuallyopenai.network.api_mesh import DecentralizedAPIMesh
        from actuallyopenai.token.aoai_token import AOAIWallet, AOAILedger, ComputeRewarder
        
        # Initialize wallet
        wallet_path = self.data_dir / "wallet.json"
        if wallet_path.exists():
            self.wallet = AOAIWallet.from_file(str(wallet_path))
            logger.info(f"💰 Wallet loaded: {self.wallet.address[:20]}...")
        else:
            self.wallet = AOAIWallet()
            self.wallet.save(str(wallet_path))
            logger.info(f"💰 New wallet created: {self.wallet.address[:20]}...")
        
        # Initialize ledger
        self.ledger = AOAILedger(str(self.data_dir / "ledger"))
        self.rewarder = ComputeRewarder(self.ledger)
        
        # Initialize P2P
        self.p2p_node = P2PNode(port=self.p2p_port, node_id=self.wallet.address)
        
        # Initialize API mesh
        self.api_mesh = DecentralizedAPIMesh(
            node_id=self.wallet.address,
            port=self.api_port
        )
        
        # Initialize scaling infrastructure
        await self._init_scaling()
    
    async def _init_scaling(self):
        """Initialize the scaling infrastructure"""
        try:
            from actuallyopenai.models.scalable_model import ScalableAOAI, ScalableConfig
            from actuallyopenai.network.compute_aggregator import NetworkComputeAggregator, ComputeCapability, NodeRole
            from actuallyopenai.orchestrator.scaling_orchestrator import ScalingOrchestrator, ScalePhase
            from actuallyopenai.training.continuous_improvement import ContinuousImprovementEngine
            
            # Detect local GPU capabilities
            gpu_memory, gpu_type = self._detect_gpu()
            
            # Create model factory
            def model_factory(scale: str):
                config = ScalableConfig.for_scale(scale)
                return ScalableAOAI(config)
            
            # Initialize scalable model
            config = ScalableConfig.for_scale(self.initial_scale)
            self.scalable_model = ScalableAOAI(config)
            
            model_params = sum(p.numel() for p in self.scalable_model.parameters())
            logger.info(f"🧠 Scalable model loaded: {self.initial_scale.upper()} ({model_params:,} params)")
            
            # Initialize compute aggregator
            role = NodeRole.COORDINATOR if gpu_memory >= 24 else NodeRole.WORKER
            self.compute_aggregator = NetworkComputeAggregator(
                node_id=self.wallet.address,
                role=role,
            )
            
            # Register our capabilities
            self.compute_aggregator.register_capabilities(ComputeCapability(
                node_id=self.wallet.address,
                gpu_memory_gb=gpu_memory,
                gpu_type=gpu_type,
                gpu_count=1,
            ))
            
            # Initialize scaling orchestrator
            self.scaling_orchestrator = ScalingOrchestrator(
                node_id=self.wallet.address,
                compute_aggregator=self.compute_aggregator,
                model_factory=model_factory,
            )
            
            # Initialize improvement engine
            self.improvement_engine = ContinuousImprovementEngine(
                model=self.scalable_model,
                min_batch_size=50,
            )
            
            logger.info(f"📈 Scaling orchestrator initialized (role: {role.value})")
            logger.info(f"🔧 GPU detected: {gpu_type} ({gpu_memory:.1f}GB)")
            
        except ImportError as e:
            logger.warning(f"Scaling modules not available: {e}")
            logger.warning("Running in basic mode without scaling")
        except Exception as e:
            logger.error(f"Failed to initialize scaling: {e}")
            import traceback
            traceback.print_exc()
    
    def _detect_gpu(self):
        """Detect GPU capabilities"""
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                return props.total_memory / (1024**3), props.name
            return 0.0, "CPU only"
        except:
            return 0.0, "Unknown"
    
    async def _start_services(self):
        """Start all services"""
        # Start P2P
        await self.p2p_node.start()
        logger.info("🌐 P2P network started")
        
        # Register inference handler
        async def chat_handler(data: dict) -> dict:
            return await self._handle_chat(data)
        
        async def scaling_status_handler(data: dict) -> dict:
            return self._get_scaling_status()
        
        self.api_mesh.register_handler("chat", chat_handler)
        self.api_mesh.register_handler("scaling_status", scaling_status_handler)
        
        # Start API
        await self.api_mesh.start()
        logger.info("🔌 API mesh started")
        
        # Start improvement loop if enabled
        if self.enable_scaling and self.improvement_engine:
            asyncio.create_task(self.improvement_engine.improvement_loop())
            logger.info("🔄 Continuous improvement engine started")
    
    async def _handle_chat(self, data: dict) -> dict:
        """Handle chat completion request"""
        import time
        import hashlib
        
        try:
            messages = data.get("messages", [])
            prompt = messages[-1]["content"] if messages else ""
            
            # Generate with scalable model
            if self.scalable_model:
                response = await self._generate(
                    prompt,
                    max_tokens=data.get("max_tokens", 256),
                    temperature=data.get("temperature", 0.7)
                )
            else:
                response = "Model not loaded"
            
            # Record for improvement (if engine available)
            if self.improvement_engine:
                self.improvement_engine.record_inference(prompt, response)
            
            return {
                "id": f"chatcmpl-{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": f"aoai-{self.initial_scale}",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": response},
                    "finish_reason": "stop"
                }]
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
        """Generate text with the scalable model"""
        import torch
        
        if self.scalable_model is None:
            return "Model not loaded"
        
        device = next(self.scalable_model.parameters()).device
        
        # Simple tokenization (use BPE tokenizer in production)
        chars = list(prompt)
        char_to_idx = {c: i for i, c in enumerate(sorted(set(chars)))}
        idx_to_char = {i: c for c, i in char_to_idx.items()}
        
        # For demo, generate character by character
        input_ids = torch.tensor([[ord(c) % 1000 for c in prompt[-512:]]], device=device)
        
        self.scalable_model.eval()
        generated = []
        
        with torch.no_grad():
            for _ in range(min(max_tokens, 256)):
                outputs = self.scalable_model(input_ids)
                
                # Handle dict output from ScalableAOAI
                if isinstance(outputs, dict):
                    logits = outputs["logits"]
                elif isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                # Get next token
                next_logits = logits[0, -1, :] / temperature
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                
                # Convert to char (simplified)
                char_idx = next_token.item() % 128
                if 32 <= char_idx < 127:
                    generated.append(chr(char_idx))
                
                # Stop on period or newline
                if generated and generated[-1] in '.!?\n':
                    break
        
        return ''.join(generated) if generated else "..."
    
    def _get_scaling_status(self) -> dict:
        """Get current scaling status"""
        status = {
            "node_id": self.wallet.address[:16] + "...",
            "current_scale": self.initial_scale,
            "model_loaded": self.scalable_model is not None,
        }
        
        if self.scalable_model:
            params = sum(p.numel() for p in self.scalable_model.parameters())
            status["model_params"] = f"{params:,}"
        
        if self.scaling_orchestrator:
            status["scaling"] = self.scaling_orchestrator.get_progress_report()
            status["roadmap"] = self.scaling_orchestrator.get_scaling_roadmap()
        
        if self.compute_aggregator:
            status["network"] = self.compute_aggregator.get_network_status()
        
        if self.improvement_engine:
            status["improvement"] = self.improvement_engine.get_stats()
        
        return status
    
    def _print_status(self):
        """Print node status"""
        stats = self.ledger.get_stats()
        balance = self.ledger.get_balance(self.wallet.address)
        
        model_params = 0
        if self.scalable_model:
            model_params = sum(p.numel() for p in self.scalable_model.parameters())
        
        gpu_mem, gpu_type = self._detect_gpu()
        
        print(f"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                     🟢 SCALING NODE RUNNING 🟢                            ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  Wallet: {self.wallet.address[:40]}...         ║
║  Balance: {balance:,.4f} AOAI                                             ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  Current Scale: {self.initial_scale.upper():8}  │  Model Params: {model_params:>14,}         ║
║  GPU: {gpu_type[:20]:20}  │  VRAM: {gpu_mem:>6.1f} GB                       ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  SCALING ROADMAP:                                                         ║
║    TINY (10M)     → SMALL (100M)    → MEDIUM (1B)                        ║
║    LARGE (7B)     → XLARGE (70B)    → FRONTIER (400B+)                   ║
║                                                                           ║
║  Network grows → Models scale → Intelligence increases                   ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  API Endpoints:                                                           ║
║    http://localhost:{self.api_port}/v1/chat/completions                         ║
║    http://localhost:{self.api_port}/scaling_status                              ║
╚═══════════════════════════════════════════════════════════════════════════╝

  🌐 Your node contributes to the global scaling effort!
  📈 As more nodes join, larger models become possible.
  🚀 Target: FRONTIER-scale intelligence through distributed compute.

  Press Ctrl+C to stop.
""")
    
    async def _run_loop(self):
        """Main run loop"""
        check_interval = 60
        
        try:
            while self.is_running:
                await asyncio.sleep(check_interval)
                
                # Log status
                balance = self.ledger.get_balance(self.wallet.address)
                peers = len(self.p2p_node.peers) if self.p2p_node else 0
                
                status_parts = [f"Balance: {balance:.4f} AOAI", f"Peers: {peers}"]
                
                if self.scaling_orchestrator:
                    report = self.scaling_orchestrator.get_progress_report()
                    status_parts.append(f"Scale: {report['current_phase']}")
                    status_parts.append(f"Progress: {report['progress']['token_progress_pct']}")
                
                if self.compute_aggregator:
                    net_status = self.compute_aggregator.get_network_status()
                    status_parts.append(f"Network: {net_status['total_nodes']} nodes")
                
                logger.info(" | ".join(status_parts))
                
                # Check if we should scale up
                if self.scaling_orchestrator:
                    should_scale, reason = self.scaling_orchestrator.should_scale_up()
                    if should_scale:
                        logger.info(f"🚀 SCALING UP: {reason}")
                        await self.scaling_orchestrator.scale_up()
                
        except asyncio.CancelledError:
            pass
    
    async def stop(self):
        """Stop the node"""
        print("\n🛑 Shutting down scaling node...")
        self.is_running = False
        
        if self.improvement_engine:
            await self.improvement_engine.stop()
        
        if self.p2p_node:
            await self.p2p_node.stop()
        
        if self.api_mesh:
            await self.api_mesh.stop()
        
        print("👋 Goodbye! Your contributions are recorded on-chain.")


async def main():
    parser = argparse.ArgumentParser(
        description="ActuallyOpenAI Scaling Node - Progressive scaling to frontier models"
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
        "--scale", type=str, default="tiny",
        choices=["tiny", "small", "medium", "large", "xlarge", "frontier"],
        help="Initial model scale (default: tiny)"
    )
    parser.add_argument(
        "--no-scaling", action="store_true",
        help="Disable automatic scaling"
    )
    
    args = parser.parse_args()
    
    node = ScalingNode(
        p2p_port=args.p2p_port,
        api_port=args.api_port,
        data_dir=args.data_dir,
        initial_scale=args.scale,
        enable_scaling=not args.no_scaling,
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
