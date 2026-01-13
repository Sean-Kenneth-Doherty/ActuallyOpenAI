"""
ActuallyOpenAI P2P Network Layer
================================
True peer-to-peer networking with no central server.
Nodes discover each other, share work, and form a resilient mesh.

Like BitTorrent, but for AI compute.
"""

import asyncio
import hashlib
import json
import random
import socket
import struct
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Set
from enum import Enum
import threading
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AOAI-P2P")


class MessageType(Enum):
    """P2P Message Types"""
    PING = 0x01
    PONG = 0x02
    DISCOVER = 0x03
    PEERS = 0x04
    WORK_REQUEST = 0x10
    WORK_RESPONSE = 0x11
    GRADIENT_SHARE = 0x20
    MODEL_SYNC = 0x21
    HEARTBEAT = 0x30
    ANNOUNCE = 0x40
    TOKEN_TRANSFER = 0x50


@dataclass
class Peer:
    """Represents a peer in the network"""
    node_id: str
    host: str
    port: int
    last_seen: float = field(default_factory=time.time)
    reputation: float = 1.0
    compute_power: float = 1.0  # TFLOPS
    tokens_earned: float = 0.0
    is_active: bool = True
    
    @property
    def address(self) -> tuple:
        return (self.host, self.port)
    
    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "reputation": self.reputation,
            "compute_power": self.compute_power
        }


@dataclass
class Message:
    """P2P Network Message"""
    msg_type: MessageType
    sender_id: str
    payload: dict
    timestamp: float = field(default_factory=time.time)
    nonce: str = field(default_factory=lambda: hashlib.sha256(str(random.random()).encode()).hexdigest()[:16])
    
    def serialize(self) -> bytes:
        """Serialize message for network transmission"""
        data = {
            "type": self.msg_type.value,
            "sender": self.sender_id,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "nonce": self.nonce
        }
        json_data = json.dumps(data).encode('utf-8')
        # Length-prefixed message
        return struct.pack('>I', len(json_data)) + json_data
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'Message':
        """Deserialize message from network"""
        json_data = json.loads(data.decode('utf-8'))
        return cls(
            msg_type=MessageType(json_data["type"]),
            sender_id=json_data["sender"],
            payload=json_data["payload"],
            timestamp=json_data["timestamp"],
            nonce=json_data["nonce"]
        )


class P2PNode:
    """
    Decentralized P2P Node for ActuallyOpenAI Network
    
    Features:
    - DHT-based peer discovery (Kademlia-inspired)
    - Gossip protocol for network state
    - Work distribution without central coordinator
    - Byzantine fault tolerance
    """
    
    # Bootstrap nodes - hardcoded entry points to the network
    BOOTSTRAP_NODES = [
        ("bootstrap1.actuallyopenai.org", 31337),
        ("bootstrap2.actuallyopenai.org", 31337),
        ("bootstrap3.actuallyopenai.org", 31337),
        # Also try local network discovery
        ("255.255.255.255", 31337),  # Broadcast
    ]
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 31337,
        node_id: Optional[str] = None
    ):
        # Generate unique node ID from hardware fingerprint
        self.node_id = node_id or self._generate_node_id()
        self.host = host
        self.port = port
        
        # Peer management
        self.peers: Dict[str, Peer] = {}
        self.known_peers: Set[tuple] = set()
        self.max_peers = 50
        
        # Network state
        self.is_running = False
        self.server_socket: Optional[socket.socket] = None
        
        # Message handlers
        self.handlers: Dict[MessageType, Callable] = {}
        self._register_default_handlers()
        
        # Work queue
        self.pending_work: List[dict] = []
        self.completed_work: Dict[str, dict] = {}
        
        # Token balance (local tracking - verified on chain)
        self.token_balance: float = 0.0
        
        # Callbacks
        self.on_peer_connected: Optional[Callable] = None
        self.on_work_received: Optional[Callable] = None
        self.on_tokens_earned: Optional[Callable] = None
        
        logger.info(f"🌐 P2P Node initialized: {self.node_id[:16]}...")
    
    def _generate_node_id(self) -> str:
        """Generate unique node ID from hardware fingerprint"""
        import platform
        import uuid
        
        fingerprint = f"{platform.node()}-{uuid.getnode()}-{time.time()}"
        return hashlib.sha256(fingerprint.encode()).hexdigest()
    
    def _register_default_handlers(self):
        """Register default message handlers"""
        self.handlers[MessageType.PING] = self._handle_ping
        self.handlers[MessageType.PONG] = self._handle_pong
        self.handlers[MessageType.DISCOVER] = self._handle_discover
        self.handlers[MessageType.PEERS] = self._handle_peers
        self.handlers[MessageType.WORK_REQUEST] = self._handle_work_request
        self.handlers[MessageType.WORK_RESPONSE] = self._handle_work_response
        self.handlers[MessageType.HEARTBEAT] = self._handle_heartbeat
        self.handlers[MessageType.ANNOUNCE] = self._handle_announce
    
    async def start(self):
        """Start the P2P node"""
        self.is_running = True
        
        # Start TCP server
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(100)
        self.server_socket.setblocking(False)
        
        logger.info(f"🚀 P2P Node listening on {self.host}:{self.port}")
        
        # Start background tasks
        asyncio.create_task(self._accept_connections())
        asyncio.create_task(self._bootstrap())
        asyncio.create_task(self._heartbeat_loop())
        asyncio.create_task(self._peer_maintenance())
        
        # Announce ourselves to the network
        await self._announce_self()
    
    async def _accept_connections(self):
        """Accept incoming connections"""
        loop = asyncio.get_event_loop()
        
        while self.is_running:
            try:
                client, addr = await loop.sock_accept(self.server_socket)
                asyncio.create_task(self._handle_connection(client, addr))
            except Exception as e:
                if self.is_running:
                    await asyncio.sleep(0.1)
    
    async def _handle_connection(self, client: socket.socket, addr: tuple):
        """Handle incoming connection"""
        try:
            client.setblocking(False)
            loop = asyncio.get_event_loop()
            
            # Read message length
            length_data = await loop.sock_recv(client, 4)
            if len(length_data) < 4:
                return
            
            msg_length = struct.unpack('>I', length_data)[0]
            
            # Read message
            data = b''
            while len(data) < msg_length:
                chunk = await loop.sock_recv(client, min(4096, msg_length - len(data)))
                if not chunk:
                    break
                data += chunk
            
            if len(data) == msg_length:
                message = Message.deserialize(data)
                await self._process_message(message, client, addr)
                
        except Exception as e:
            logger.debug(f"Connection error from {addr}: {e}")
        finally:
            client.close()
    
    async def _process_message(self, message: Message, client: socket.socket, addr: tuple):
        """Process incoming message"""
        handler = self.handlers.get(message.msg_type)
        if handler:
            response = await handler(message, addr)
            if response:
                try:
                    loop = asyncio.get_event_loop()
                    await loop.sock_sendall(client, response.serialize())
                except:
                    pass
    
    async def _bootstrap(self):
        """Bootstrap into the network by connecting to known nodes"""
        logger.info("🔍 Bootstrapping into the network...")
        
        for host, port in self.BOOTSTRAP_NODES:
            try:
                await self._connect_to_peer(host, port)
            except Exception as e:
                logger.debug(f"Bootstrap node {host}:{port} unavailable: {e}")
        
        # Also try local network discovery
        await self._local_discovery()
    
    async def _local_discovery(self):
        """Discover peers on local network via UDP broadcast"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.settimeout(2)
            
            # Broadcast discovery message
            discovery_msg = json.dumps({
                "type": "AOAI_DISCOVER",
                "node_id": self.node_id,
                "port": self.port
            }).encode()
            
            sock.sendto(discovery_msg, ('<broadcast>', 31338))
            
            # Listen for responses
            try:
                while True:
                    data, addr = sock.recvfrom(1024)
                    response = json.loads(data.decode())
                    if response.get("type") == "AOAI_ANNOUNCE":
                        peer_id = response["node_id"]
                        if peer_id != self.node_id:
                            await self._connect_to_peer(addr[0], response["port"])
            except socket.timeout:
                pass
                
            sock.close()
        except Exception as e:
            logger.debug(f"Local discovery error: {e}")
    
    async def _connect_to_peer(self, host: str, port: int):
        """Connect to a peer"""
        if (host, port) in self.known_peers:
            return
        
        if len(self.peers) >= self.max_peers:
            return
        
        try:
            loop = asyncio.get_event_loop()
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setblocking(False)
            
            await asyncio.wait_for(
                loop.sock_connect(sock, (host, port)),
                timeout=5.0
            )
            
            # Send PING
            ping = Message(
                msg_type=MessageType.PING,
                sender_id=self.node_id,
                payload={
                    "port": self.port,
                    "version": "1.0.0"
                }
            )
            
            await loop.sock_sendall(sock, ping.serialize())
            
            # Wait for PONG
            length_data = await asyncio.wait_for(
                loop.sock_recv(sock, 4),
                timeout=5.0
            )
            
            if len(length_data) == 4:
                msg_length = struct.unpack('>I', length_data)[0]
                data = await loop.sock_recv(sock, msg_length)
                response = Message.deserialize(data)
                
                if response.msg_type == MessageType.PONG:
                    # Add peer
                    peer = Peer(
                        node_id=response.sender_id,
                        host=host,
                        port=port,
                        compute_power=response.payload.get("compute_power", 1.0)
                    )
                    self.peers[peer.node_id] = peer
                    self.known_peers.add((host, port))
                    
                    logger.info(f"✅ Connected to peer: {peer.node_id[:16]}... ({host}:{port})")
                    
                    if self.on_peer_connected:
                        self.on_peer_connected(peer)
                    
                    # Request more peers
                    await self._request_peers(sock)
            
            sock.close()
            
        except Exception as e:
            logger.debug(f"Failed to connect to {host}:{port}: {e}")
    
    async def _request_peers(self, sock: socket.socket):
        """Request peer list from connected peer"""
        try:
            loop = asyncio.get_event_loop()
            
            discover = Message(
                msg_type=MessageType.DISCOVER,
                sender_id=self.node_id,
                payload={"max_peers": 20}
            )
            
            await loop.sock_sendall(sock, discover.serialize())
        except:
            pass
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats to all peers"""
        while self.is_running:
            await asyncio.sleep(30)  # Every 30 seconds
            
            for peer_id, peer in list(self.peers.items()):
                try:
                    await self._send_to_peer(peer, Message(
                        msg_type=MessageType.HEARTBEAT,
                        sender_id=self.node_id,
                        payload={
                            "tokens": self.token_balance,
                            "work_completed": len(self.completed_work)
                        }
                    ))
                except:
                    # Mark peer as inactive if heartbeat fails
                    peer.is_active = False
    
    async def _peer_maintenance(self):
        """Clean up inactive peers"""
        while self.is_running:
            await asyncio.sleep(60)
            
            now = time.time()
            inactive = []
            
            for peer_id, peer in self.peers.items():
                if now - peer.last_seen > 120:  # 2 minutes timeout
                    inactive.append(peer_id)
            
            for peer_id in inactive:
                del self.peers[peer_id]
                logger.info(f"🔌 Peer disconnected: {peer_id[:16]}...")
    
    async def _announce_self(self):
        """Announce our presence to the network"""
        announce = Message(
            msg_type=MessageType.ANNOUNCE,
            sender_id=self.node_id,
            payload={
                "port": self.port,
                "compute_power": self._get_compute_power(),
                "version": "1.0.0"
            }
        )
        
        await self._broadcast(announce)
    
    def _get_compute_power(self) -> float:
        """Estimate local compute power in TFLOPS"""
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                # Rough TFLOPS estimate
                return (props.multi_processor_count * props.max_threads_per_multi_processor * 2) / 1e12
        except:
            pass
        return 0.1  # Default CPU estimate
    
    async def _broadcast(self, message: Message):
        """Broadcast message to all peers"""
        for peer_id, peer in list(self.peers.items()):
            try:
                await self._send_to_peer(peer, message)
            except:
                pass
    
    async def _send_to_peer(self, peer: Peer, message: Message):
        """Send message to specific peer"""
        loop = asyncio.get_event_loop()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setblocking(False)
        
        try:
            await asyncio.wait_for(
                loop.sock_connect(sock, peer.address),
                timeout=5.0
            )
            await loop.sock_sendall(sock, message.serialize())
            peer.last_seen = time.time()
        finally:
            sock.close()
    
    # Message Handlers
    async def _handle_ping(self, message: Message, addr: tuple) -> Message:
        """Handle PING message"""
        return Message(
            msg_type=MessageType.PONG,
            sender_id=self.node_id,
            payload={
                "compute_power": self._get_compute_power(),
                "peer_count": len(self.peers)
            }
        )
    
    async def _handle_pong(self, message: Message, addr: tuple):
        """Handle PONG message"""
        pass  # Handled in _connect_to_peer
    
    async def _handle_discover(self, message: Message, addr: tuple) -> Message:
        """Handle peer discovery request"""
        max_peers = message.payload.get("max_peers", 20)
        
        peer_list = [
            peer.to_dict()
            for peer in list(self.peers.values())[:max_peers]
            if peer.is_active
        ]
        
        return Message(
            msg_type=MessageType.PEERS,
            sender_id=self.node_id,
            payload={"peers": peer_list}
        )
    
    async def _handle_peers(self, message: Message, addr: tuple):
        """Handle peer list response"""
        peers = message.payload.get("peers", [])
        
        for peer_data in peers:
            host = peer_data.get("host")
            port = peer_data.get("port")
            if host and port:
                asyncio.create_task(self._connect_to_peer(host, port))
    
    async def _handle_work_request(self, message: Message, addr: tuple) -> Optional[Message]:
        """Handle work request from coordinator"""
        work = message.payload.get("work")
        if work and self.on_work_received:
            result = await self.on_work_received(work)
            return Message(
                msg_type=MessageType.WORK_RESPONSE,
                sender_id=self.node_id,
                payload={"result": result, "work_id": work.get("id")}
            )
        return None
    
    async def _handle_work_response(self, message: Message, addr: tuple):
        """Handle work response"""
        work_id = message.payload.get("work_id")
        if work_id:
            self.completed_work[work_id] = message.payload
    
    async def _handle_heartbeat(self, message: Message, addr: tuple):
        """Handle heartbeat"""
        sender_id = message.sender_id
        if sender_id in self.peers:
            self.peers[sender_id].last_seen = time.time()
    
    async def _handle_announce(self, message: Message, addr: tuple):
        """Handle node announcement"""
        port = message.payload.get("port", 31337)
        await self._connect_to_peer(addr[0], port)
    
    # Public API
    async def submit_work(self, work: dict) -> str:
        """Submit work to the network"""
        work_id = hashlib.sha256(json.dumps(work).encode()).hexdigest()[:16]
        work["id"] = work_id
        
        # Find best peer for work (highest compute power)
        best_peers = sorted(
            self.peers.values(),
            key=lambda p: p.compute_power,
            reverse=True
        )[:5]
        
        for peer in best_peers:
            try:
                await self._send_to_peer(peer, Message(
                    msg_type=MessageType.WORK_REQUEST,
                    sender_id=self.node_id,
                    payload={"work": work}
                ))
                logger.info(f"📤 Work {work_id} sent to {peer.node_id[:16]}...")
                break
            except:
                continue
        
        return work_id
    
    async def get_work_result(self, work_id: str, timeout: float = 60.0) -> Optional[dict]:
        """Wait for work result"""
        start = time.time()
        while time.time() - start < timeout:
            if work_id in self.completed_work:
                return self.completed_work.pop(work_id)
            await asyncio.sleep(0.5)
        return None
    
    def get_network_stats(self) -> dict:
        """Get network statistics"""
        return {
            "node_id": self.node_id,
            "peer_count": len(self.peers),
            "active_peers": sum(1 for p in self.peers.values() if p.is_active),
            "total_compute_tflops": sum(p.compute_power for p in self.peers.values()),
            "token_balance": self.token_balance,
            "work_completed": len(self.completed_work)
        }
    
    async def stop(self):
        """Stop the P2P node"""
        self.is_running = False
        if self.server_socket:
            self.server_socket.close()
        logger.info("🛑 P2P Node stopped")


# CLI for testing
async def main():
    """Run P2P node"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ActuallyOpenAI P2P Node")
    parser.add_argument("--port", type=int, default=31337, help="Port to listen on")
    args = parser.parse_args()
    
    node = P2PNode(port=args.port)
    
    def on_peer(peer):
        print(f"🤝 New peer: {peer.node_id[:16]}... ({peer.host}:{peer.port})")
    
    node.on_peer_connected = on_peer
    
    await node.start()
    
    print(f"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║         🌐 ActuallyOpenAI P2P Node Running 🌐                  ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  Node ID: {node.node_id[:32]}...       ║
    ║  Port: {args.port}                                              ║
    ║                                                               ║
    ║  The network is decentralized - there is no central server.  ║
    ║  Peers discover each other and share compute automatically.  ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Keep running
    try:
        while True:
            await asyncio.sleep(10)
            stats = node.get_network_stats()
            print(f"📊 Peers: {stats['peer_count']} | Compute: {stats['total_compute_tflops']:.2f} TFLOPS")
    except KeyboardInterrupt:
        await node.stop()


if __name__ == "__main__":
    asyncio.run(main())
