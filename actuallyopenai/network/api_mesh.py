"""
Decentralized API Mesh
======================
No single API server. Every node can serve requests.
Requests are routed to the best available node automatically.

Like a CDN, but for AI inference - and completely decentralized.
"""

import asyncio
import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Set
from enum import Enum
import logging
import aiohttp
from aiohttp import web

# Local imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger("AOAI-Mesh")


@dataclass
class APINode:
    """Represents an API node in the mesh"""
    node_id: str
    host: str
    port: int
    capacity: float = 1.0       # Requests per second
    latency_ms: float = 100.0   # Average response time
    success_rate: float = 1.0   # Success rate (0-1)
    is_healthy: bool = True
    last_check: float = field(default_factory=time.time)
    specializations: List[str] = field(default_factory=list)  # e.g., ["chat", "embeddings"]
    
    @property
    def score(self) -> float:
        """Calculate node score for routing"""
        if not self.is_healthy:
            return 0
        return (self.capacity * self.success_rate) / (self.latency_ms + 1)
    
    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


class LoadBalancer:
    """
    Intelligent load balancer for the mesh
    
    Routes requests to the best available node based on:
    - Latency
    - Capacity
    - Success rate
    - Specialization
    """
    
    def __init__(self):
        self.nodes: Dict[str, APINode] = {}
        self.request_counts: Dict[str, int] = {}
        self.error_counts: Dict[str, int] = {}
    
    def add_node(self, node: APINode):
        """Add a node to the load balancer"""
        self.nodes[node.node_id] = node
        self.request_counts[node.node_id] = 0
        self.error_counts[node.node_id] = 0
        logger.info(f"➕ Node added: {node.node_id[:16]}... ({node.host}:{node.port})")
    
    def remove_node(self, node_id: str):
        """Remove a node from the load balancer"""
        if node_id in self.nodes:
            del self.nodes[node_id]
            logger.info(f"➖ Node removed: {node_id[:16]}...")
    
    def get_best_node(self, specialization: Optional[str] = None) -> Optional[APINode]:
        """Get the best available node"""
        candidates = [
            node for node in self.nodes.values()
            if node.is_healthy
        ]
        
        # Filter by specialization if specified
        if specialization:
            specialized = [
                n for n in candidates
                if specialization in n.specializations or not n.specializations
            ]
            if specialized:
                candidates = specialized
        
        if not candidates:
            return None
        
        # Weighted random selection based on score
        total_score = sum(n.score for n in candidates)
        if total_score == 0:
            return random.choice(candidates)
        
        r = random.random() * total_score
        cumulative = 0
        for node in candidates:
            cumulative += node.score
            if r <= cumulative:
                return node
        
        return candidates[-1]
    
    def record_success(self, node_id: str, latency_ms: float):
        """Record successful request"""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            # Exponential moving average for latency
            node.latency_ms = 0.9 * node.latency_ms + 0.1 * latency_ms
            self.request_counts[node_id] += 1
            # Update success rate
            total = self.request_counts[node_id]
            errors = self.error_counts.get(node_id, 0)
            node.success_rate = (total - errors) / total
    
    def record_error(self, node_id: str):
        """Record failed request"""
        self.error_counts[node_id] = self.error_counts.get(node_id, 0) + 1
        if node_id in self.nodes:
            total = self.request_counts.get(node_id, 0) + 1
            errors = self.error_counts[node_id]
            self.nodes[node_id].success_rate = (total - errors) / total
            
            # Mark unhealthy if too many errors
            if self.nodes[node_id].success_rate < 0.5:
                self.nodes[node_id].is_healthy = False
                logger.warning(f"⚠️ Node marked unhealthy: {node_id[:16]}...")


class DecentralizedAPIMesh:
    """
    Decentralized API mesh for ActuallyOpenAI
    
    Features:
    - No single point of failure
    - Automatic load balancing
    - Self-healing (unhealthy nodes removed)
    - Geographic distribution
    - Request deduplication
    """
    
    def __init__(
        self,
        node_id: Optional[str] = None,
        host: str = "0.0.0.0",
        port: int = 8080
    ):
        self.node_id = node_id or hashlib.sha256(
            f"{host}:{port}:{time.time()}".encode()
        ).hexdigest()
        
        self.host = host
        self.port = port
        
        # Load balancer for other nodes
        self.load_balancer = LoadBalancer()
        
        # Request cache (deduplication)
        self.request_cache: Dict[str, dict] = {}
        self.cache_ttl = 60  # seconds
        
        # Health check interval
        self.health_check_interval = 30  # seconds
        
        # API handlers
        self.handlers: Dict[str, Callable] = {}
        
        # External stats provider (for adaptive miner etc)
        self.stats_provider: Optional[Callable] = None
        
        # Web app
        self.app = web.Application()
        self._setup_routes()
        
        # Running state
        self.is_running = False
        
        logger.info(f"🌐 API Mesh node initialized: {self.node_id[:16]}...")
    
    def _setup_routes(self):
        """Setup API routes"""
        self.app.router.add_get('/health', self._handle_health)
        self.app.router.add_post('/v1/chat/completions', self._handle_chat)
        self.app.router.add_post('/v1/embeddings', self._handle_embeddings)
        self.app.router.add_post('/v1/completions', self._handle_completions)
        self.app.router.add_get('/v1/models', self._handle_models)
        self.app.router.add_get('/mesh/nodes', self._handle_list_nodes)
        self.app.router.add_post('/mesh/join', self._handle_join)
        self.app.router.add_get('/mesh/stats', self._handle_stats)
    
    async def start(self):
        """Start the API mesh node"""
        self.is_running = True
        
        # Start health check loop
        asyncio.create_task(self._health_check_loop())
        
        # Start cache cleanup loop
        asyncio.create_task(self._cache_cleanup_loop())
        
        # Start web server
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        
        logger.info(f"🚀 API Mesh running at http://{self.host}:{self.port}")
        
        return runner
    
    async def _health_check_loop(self):
        """Periodically check health of other nodes"""
        while self.is_running:
            await asyncio.sleep(self.health_check_interval)
            
            for node_id, node in list(self.load_balancer.nodes.items()):
                try:
                    async with aiohttp.ClientSession() as session:
                        start = time.time()
                        async with session.get(
                            f"{node.url}/health",
                            timeout=aiohttp.ClientTimeout(total=5)
                        ) as response:
                            if response.status == 200:
                                latency = (time.time() - start) * 1000
                                node.is_healthy = True
                                node.latency_ms = latency
                                node.last_check = time.time()
                            else:
                                node.is_healthy = False
                except Exception as e:
                    node.is_healthy = False
                    logger.debug(f"Health check failed for {node_id[:16]}: {e}")
    
    async def _cache_cleanup_loop(self):
        """Clean up expired cache entries"""
        while self.is_running:
            await asyncio.sleep(self.cache_ttl)
            
            now = time.time()
            expired = [
                key for key, value in self.request_cache.items()
                if now - value.get("timestamp", 0) > self.cache_ttl
            ]
            for key in expired:
                del self.request_cache[key]
    
    def _get_cache_key(self, request_data: dict) -> str:
        """Generate cache key for request deduplication"""
        content = json.dumps(request_data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    async def _forward_request(
        self,
        endpoint: str,
        data: dict,
        method: str = "POST"
    ) -> Optional[dict]:
        """Forward request to another node in the mesh"""
        node = self.load_balancer.get_best_node()
        
        if not node:
            return None
        
        try:
            async with aiohttp.ClientSession() as session:
                start = time.time()
                
                if method == "POST":
                    async with session.post(
                        f"{node.url}{endpoint}",
                        json=data,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        latency = (time.time() - start) * 1000
                        
                        if response.status == 200:
                            self.load_balancer.record_success(node.node_id, latency)
                            return await response.json()
                        else:
                            self.load_balancer.record_error(node.node_id)
                else:
                    async with session.get(
                        f"{node.url}{endpoint}",
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        latency = (time.time() - start) * 1000
                        
                        if response.status == 200:
                            self.load_balancer.record_success(node.node_id, latency)
                            return await response.json()
                        else:
                            self.load_balancer.record_error(node.node_id)
                            
        except Exception as e:
            self.load_balancer.record_error(node.node_id)
            logger.error(f"Forward to {node.node_id[:16]} failed: {e}")
        
        return None
    
    # API Handlers
    async def _handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint"""
        return web.json_response({
            "status": "healthy",
            "node_id": self.node_id,
            "mesh_nodes": len(self.load_balancer.nodes),
            "timestamp": time.time()
        })
    
    async def _handle_chat(self, request: web.Request) -> web.Response:
        """Handle chat completions"""
        try:
            data = await request.json()
            
            # Check cache
            cache_key = self._get_cache_key(data)
            if cache_key in self.request_cache:
                logger.debug("Cache hit!")
                return web.json_response(self.request_cache[cache_key]["response"])
            
            # Try local handler first
            if "chat" in self.handlers:
                response = await self.handlers["chat"](data)
            else:
                # Forward to mesh
                response = await self._forward_request("/v1/chat/completions", data)
            
            if response:
                # Cache response
                self.request_cache[cache_key] = {
                    "response": response,
                    "timestamp": time.time()
                }
                return web.json_response(response)
            
            return web.json_response(
                {"error": "No available nodes"},
                status=503
            )
            
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return web.json_response(
                {"error": str(e)},
                status=500
            )
    
    async def _handle_embeddings(self, request: web.Request) -> web.Response:
        """Handle embeddings"""
        try:
            data = await request.json()
            
            if "embeddings" in self.handlers:
                response = await self.handlers["embeddings"](data)
            else:
                response = await self._forward_request("/v1/embeddings", data)
            
            if response:
                return web.json_response(response)
            
            return web.json_response(
                {"error": "No available nodes"},
                status=503
            )
            
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
    
    async def _handle_completions(self, request: web.Request) -> web.Response:
        """Handle text completions"""
        try:
            data = await request.json()
            
            if "completions" in self.handlers:
                response = await self.handlers["completions"](data)
            else:
                response = await self._forward_request("/v1/completions", data)
            
            if response:
                return web.json_response(response)
            
            return web.json_response(
                {"error": "No available nodes"},
                status=503
            )
            
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
    
    async def _handle_models(self, request: web.Request) -> web.Response:
        """List available models"""
        return web.json_response({
            "object": "list",
            "data": [
                {
                    "id": "aoai-1",
                    "object": "model",
                    "created": 1704067200,
                    "owned_by": "actuallyopenai-network",
                    "permission": [],
                    "root": "aoai-1",
                    "parent": None
                },
                {
                    "id": "aoai-embed",
                    "object": "model",
                    "created": 1704067200,
                    "owned_by": "actuallyopenai-network"
                }
            ]
        })
    
    async def _handle_list_nodes(self, request: web.Request) -> web.Response:
        """List all nodes in the mesh"""
        nodes = []
        for node_id, node in self.load_balancer.nodes.items():
            nodes.append({
                "node_id": node_id,
                "host": node.host,
                "port": node.port,
                "is_healthy": node.is_healthy,
                "latency_ms": node.latency_ms,
                "success_rate": node.success_rate,
                "score": node.score
            })
        
        return web.json_response({
            "self": {
                "node_id": self.node_id,
                "host": self.host,
                "port": self.port
            },
            "nodes": nodes,
            "total": len(nodes)
        })
    
    async def _handle_join(self, request: web.Request) -> web.Response:
        """Handle node join request"""
        try:
            data = await request.json()
            
            node = APINode(
                node_id=data.get("node_id", hashlib.sha256(str(time.time()).encode()).hexdigest()),
                host=data["host"],
                port=data["port"],
                specializations=data.get("specializations", [])
            )
            
            self.load_balancer.add_node(node)
            
            return web.json_response({
                "status": "joined",
                "node_id": node.node_id,
                "mesh_size": len(self.load_balancer.nodes) + 1
            })
            
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)
    
    async def _handle_stats(self, request: web.Request) -> web.Response:
        """Get mesh statistics"""
        healthy_nodes = sum(1 for n in self.load_balancer.nodes.values() if n.is_healthy)
        total_capacity = sum(n.capacity for n in self.load_balancer.nodes.values() if n.is_healthy)
        avg_latency = 0
        if self.load_balancer.nodes:
            avg_latency = sum(n.latency_ms for n in self.load_balancer.nodes.values()) / len(self.load_balancer.nodes)
        
        stats = {
            "total_nodes": len(self.load_balancer.nodes) + 1,  # +1 for self
            "healthy_nodes": healthy_nodes + 1,
            "total_capacity_rps": total_capacity,
            "avg_latency_ms": avg_latency,
            "cache_entries": len(self.request_cache),
            "uptime": time.time()
        }
        
        # Add external stats (e.g., adaptive miner)
        if self.stats_provider:
            try:
                extra_stats = self.stats_provider()
                if extra_stats:
                    stats["miner"] = extra_stats
            except Exception as e:
                logger.debug(f"Stats provider error: {e}")
        
        return web.json_response(stats)
    
    def register_handler(self, endpoint: str, handler: Callable):
        """Register a local handler for an endpoint"""
        self.handlers[endpoint] = handler
        logger.info(f"📝 Handler registered: {endpoint}")
    
    async def join_mesh(self, bootstrap_url: str):
        """Join an existing mesh"""
        try:
            async with aiohttp.ClientSession() as session:
                # Register with bootstrap node
                async with session.post(
                    f"{bootstrap_url}/mesh/join",
                    json={
                        "node_id": self.node_id,
                        "host": self.host,
                        "port": self.port,
                        "specializations": list(self.handlers.keys())
                    }
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"✅ Joined mesh! Size: {result['mesh_size']}")
                
                # Get list of other nodes
                async with session.get(f"{bootstrap_url}/mesh/nodes") as response:
                    if response.status == 200:
                        data = await response.json()
                        for node_data in data.get("nodes", []):
                            node = APINode(
                                node_id=node_data["node_id"],
                                host=node_data["host"],
                                port=node_data["port"]
                            )
                            self.load_balancer.add_node(node)
                        
                        # Add bootstrap node too
                        bootstrap_info = data.get("self", {})
                        if bootstrap_info:
                            self.load_balancer.add_node(APINode(
                                node_id=bootstrap_info.get("node_id", "bootstrap"),
                                host=bootstrap_info.get("host", "localhost"),
                                port=bootstrap_info.get("port", 8080)
                            ))
                            
        except Exception as e:
            logger.error(f"Failed to join mesh: {e}")
    
    async def stop(self):
        """Stop the mesh node"""
        self.is_running = False
        logger.info("🛑 API Mesh node stopped")


# Entrypoint for running a mesh node
async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="ActuallyOpenAI API Mesh Node")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--join", help="Bootstrap URL to join existing mesh")
    args = parser.parse_args()
    
    mesh = DecentralizedAPIMesh(host=args.host, port=args.port)
    
    # Register a simple local handler for demo
    async def local_chat_handler(data: dict) -> dict:
        messages = data.get("messages", [])
        last_message = messages[-1]["content"] if messages else "Hello"
        
        return {
            "id": f"chatcmpl-{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "aoai-1",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"[Mesh Node {mesh.node_id[:8]}] Received: {last_message}"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(last_message.split()),
                "completion_tokens": 10,
                "total_tokens": len(last_message.split()) + 10
            }
        }
    
    mesh.register_handler("chat", local_chat_handler)
    
    # Start the mesh
    runner = await mesh.start()
    
    # Join existing mesh if specified
    if args.join:
        await mesh.join_mesh(args.join)
    
    print(f"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║         🌐 ActuallyOpenAI API Mesh Node Running 🌐            ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  Node ID: {mesh.node_id[:32]}...       ║
    ║  URL: http://{args.host}:{args.port}                               ║
    ║                                                               ║
    ║  Endpoints:                                                   ║
    ║    POST /v1/chat/completions  - Chat API                      ║
    ║    POST /v1/embeddings        - Embeddings API                ║
    ║    GET  /v1/models            - List models                   ║
    ║    GET  /mesh/nodes           - List mesh nodes               ║
    ║    GET  /mesh/stats           - Mesh statistics               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Keep running
    try:
        while True:
            await asyncio.sleep(3600)
    except KeyboardInterrupt:
        await mesh.stop()
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
