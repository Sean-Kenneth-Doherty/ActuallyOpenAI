#!/usr/bin/env python3
"""
ActuallyOpenAI - Decentralized AI Node
======================================
Run this to join the network and start contributing.

Your computer becomes part of a global, decentralized AI.
Earn AOAI tokens for your contribution.

Usage:
    python node.py                    # Start with defaults
    python node.py --p2p-port 31337   # Custom P2P port
    python node.py --api-port 8080    # Custom API port
    python node.py --no-mining        # Don't contribute compute
"""

import asyncio
import argparse
import logging
import signal
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger("AOAI")


BANNER = """
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
║                    🌐 THE PEOPLE'S AI - DECENTRALIZED 🌐                  ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""


class AOAINode:
    """
    Main node that runs all decentralized components:
    - P2P networking for peer discovery
    - IPFS for model storage
    - Token system for rewards
    - API mesh for inference
    """
    
    def __init__(
        self,
        p2p_port: int = 31337,
        api_port: int = 8080,
        data_dir: str = "./aoai_data",
        enable_mining: bool = True
    ):
        self.p2p_port = p2p_port
        self.api_port = api_port
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.enable_mining = enable_mining
        
        # Components (lazy loaded)
        self.p2p_node = None
        self.api_mesh = None
        self.wallet = None
        self.ledger = None
        self.ipfs_storage = None
        self.rewarder = None
        
        self.is_running = False
    
    async def start(self):
        """Start all node components"""
        print(BANNER)
        print("🚀 Starting ActuallyOpenAI Node...\n")
        
        self.is_running = True
        
        # Initialize components
        await self._init_components()
        
        # Start P2P networking
        await self._start_p2p()
        
        # Start API mesh
        await self._start_api_mesh()
        
        # Start mining if enabled
        if self.enable_mining:
            asyncio.create_task(self._mining_loop())
        
        # Print status
        self._print_status()
        
        # Keep running
        await self._run_forever()
    
    async def _init_components(self):
        """Initialize all components"""
        from actuallyopenai.network.p2p_node import P2PNode
        from actuallyopenai.network.ipfs_storage import IPFSModelStorage
        from actuallyopenai.network.api_mesh import DecentralizedAPIMesh
        from actuallyopenai.token.aoai_token import AOAIWallet, AOAILedger, ComputeRewarder
        
        # Create or load wallet
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
        
        # Initialize rewarder
        self.rewarder = ComputeRewarder(self.ledger)
        
        # Initialize IPFS storage
        self.ipfs_storage = IPFSModelStorage(
            str(self.data_dir / "ipfs_cache")
        )
        
        # Initialize P2P node
        self.p2p_node = P2PNode(port=self.p2p_port, node_id=self.wallet.address)
        
        # Initialize API mesh
        self.api_mesh = DecentralizedAPIMesh(
            node_id=self.wallet.address,
            port=self.api_port
        )
    
    async def _start_p2p(self):
        """Start P2P networking"""
        logger.info("🌐 Starting P2P network...")
        
        # Set up callbacks
        def on_peer_connected(peer):
            logger.info(f"🤝 Peer connected: {peer.node_id[:16]}... ({peer.host}:{peer.port})")
        
        async def on_work_received(work):
            # Process work and earn tokens
            logger.info(f"⚡ Received work: {work.get('id', 'unknown')}")
            
            # Simulate processing
            await asyncio.sleep(1)
            
            # Record work and get reward
            reward = self.rewarder.record_work(
                self.wallet.address,
                work_type=work.get("type", "inference"),
                compute_units=work.get("compute_units", 1.0),
                proof={"work_id": work.get("id"), "timestamp": __import__('time').time()}
            )
            
            return {"status": "completed", "reward": reward}
        
        self.p2p_node.on_peer_connected = on_peer_connected
        self.p2p_node.on_work_received = on_work_received
        
        await self.p2p_node.start()
    
    async def _start_api_mesh(self):
        """Start API mesh"""
        logger.info("🔌 Starting API mesh...")
        
        # Register local inference handler
        async def chat_handler(data: dict) -> dict:
            try:
                # Try to use local model
                from actuallyopenai.api.model_inference import ModelInference
                inference = ModelInference()
                
                messages = data.get("messages", [])
                prompt = messages[-1]["content"] if messages else ""
                
                response = inference.generate(
                    prompt,
                    max_tokens=data.get("max_tokens", 256),
                    temperature=data.get("temperature", 0.7)
                )
                
                # Record work for tokens
                self.rewarder.record_work(
                    self.wallet.address,
                    work_type="inference",
                    compute_units=0.1,
                    proof={"prompt_length": len(prompt)}
                )
                
                return {
                    "id": f"chatcmpl-{__import__('hashlib').sha256(str(__import__('time').time()).encode()).hexdigest()[:8]}",
                    "object": "chat.completion",
                    "created": int(__import__('time').time()),
                    "model": "aoai-1",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": response},
                        "finish_reason": "stop"
                    }]
                }
            except Exception as e:
                logger.error(f"Inference error: {e}")
                return {"error": str(e)}
        
        self.api_mesh.register_handler("chat", chat_handler)
        
        await self.api_mesh.start()
    
    async def _mining_loop(self):
        """Mine blocks and process rewards"""
        logger.info("⛏️ Mining enabled - contributing compute...")
        
        while self.is_running:
            try:
                # Mine pending transactions
                block = self.ledger.mine_block(self.wallet.address)
                if block:
                    logger.info(f"⛏️ Mined block #{block.index}!")
                
                await asyncio.sleep(10)  # Mine every 10 seconds
                
            except Exception as e:
                logger.error(f"Mining error: {e}")
                await asyncio.sleep(30)
    
    def _print_status(self):
        """Print node status"""
        stats = self.ledger.get_stats()
        balance = self.ledger.get_balance(self.wallet.address)
        
        print(f"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                        🟢 NODE RUNNING 🟢                                 ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  Wallet: {self.wallet.address[:40]}...         ║
║  Balance: {balance:,.4f} AOAI                                             ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  P2P Port: {self.p2p_port}      │  API Port: {self.api_port}                              ║
║  Mining: {'ON' if self.enable_mining else 'OFF'}          │  Peers: {len(self.p2p_node.peers) if self.p2p_node else 0}                                     ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  Network Stats:                                                           ║
║    Total Minted: {stats['total_minted']:>12,.2f} AOAI                                    ║
║    Block Height: {stats['blocks']:>12}                                               ║
║    Block Reward: {stats['current_reward']:>12.4f} AOAI                                    ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  API Endpoints:                                                           ║
║    http://localhost:{self.api_port}/v1/chat/completions                         ║
║    http://localhost:{self.api_port}/v1/embeddings                               ║
║    http://localhost:{self.api_port}/mesh/stats                                  ║
╚═══════════════════════════════════════════════════════════════════════════╝

  📖 Your node is now part of the decentralized AI network!
  💰 You earn AOAI tokens for every computation you contribute.
  🌐 The more you contribute, the more you earn.

  Press Ctrl+C to stop.
""")
    
    async def _run_forever(self):
        """Keep the node running"""
        try:
            while self.is_running:
                await asyncio.sleep(60)
                
                # Print periodic stats
                balance = self.ledger.get_balance(self.wallet.address)
                peers = len(self.p2p_node.peers) if self.p2p_node else 0
                logger.info(f"📊 Balance: {balance:.4f} AOAI | Peers: {peers}")
                
        except asyncio.CancelledError:
            pass
    
    async def stop(self):
        """Stop all node components"""
        print("\n🛑 Shutting down node...")
        self.is_running = False
        
        if self.p2p_node:
            await self.p2p_node.stop()
        
        if self.api_mesh:
            await self.api_mesh.stop()
        
        print("👋 Goodbye! Your tokens are safe in your wallet.")


async def main():
    parser = argparse.ArgumentParser(
        description="ActuallyOpenAI Decentralized Node"
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
        "--no-mining", action="store_true",
        help="Disable compute contribution"
    )
    
    args = parser.parse_args()
    
    node = AOAINode(
        p2p_port=args.p2p_port,
        api_port=args.api_port,
        data_dir=args.data_dir,
        enable_mining=not args.no_mining
    )
    
    # Handle shutdown gracefully
    loop = asyncio.get_event_loop()
    
    def signal_handler():
        asyncio.create_task(node.stop())
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass
    
    try:
        await node.start()
    except KeyboardInterrupt:
        await node.stop()


if __name__ == "__main__":
    asyncio.run(main())
