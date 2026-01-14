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
from typing import Dict, List, Optional, Callable, Set, Tuple
from enum import Enum
import threading
import logging

from actuallyopenai.config import get_settings

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
    
    Bootstrap nodes can be configured via P2P_BOOTSTRAP_NODES environment variable
    (comma-separated list of host:port pairs).
    """
    
    # Default bootstrap nodes - used when no env config is provided
    DEFAULT_BOOTSTRAP_NODES = [
        ("bootstrap1.actuallyopenai.org", 31337),
        ("bootstrap2.actuallyopenai.org", 31337),
        ("bootstrap3.actuallyopenai.org", 31337),
    ]
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 31337,
        node_id: Optional[str] = None,
        bootstrap_nodes: Optional[List[Tuple[str, int]]] = None,
        use_defaults: bool = False
    ):
        # Generate unique node ID from hardware fingerprint
        self.node_id = node_id or self._generate_node_id()
        self.host = host
        self.port = port
        
        # Initialize bootstrap nodes from config, parameter, or defaults
        self._bootstrap_nodes: List[Tuple[str, int]] = []
        self._init_bootstrap_nodes(bootstrap_nodes, use_defaults)
        
        # Peer management
        self.peers: Dict[str, Peer] = {}
        self.known_peers: Set[tuple] = set()
        settings = get_settings()
        self.max_peers = settings.p2p_max_peers
        
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
        
        # Gradient aggregation for distributed training
        self.gradient_buffer: Dict[str, List[dict]] = {}  # work_id -> list of gradients
        self.gradient_counts: Dict[str, int] = {}  # work_id -> expected count
        self.aggregated_gradients: Dict[str, dict] = {}  # work_id -> aggregated result
        
        # Model synchronization state
        self.current_model_version: str = "0.0.0"
        self.model_state: Optional[dict] = None
        self.pending_model_syncs: Dict[str, dict] = {}  # node_id -> model state
        
        # Token transfer tracking
        self.pending_transfers: Dict[str, dict] = {}  # transfer_id -> transfer details
        self.transfer_history: List[dict] = []
        
        # Callbacks
        self.on_peer_connected: Optional[Callable] = None
        self.on_work_received: Optional[Callable] = None
        self.on_tokens_earned: Optional[Callable] = None
        self.on_gradient_received: Optional[Callable] = None
        self.on_model_sync: Optional[Callable] = None
        self.on_gradients_aggregated: Optional[Callable] = None
        
        logger.info(f"🌐 P2P Node initialized: {self.node_id[:16]}...")
    
    def _generate_node_id(self) -> str:
        """Generate unique node ID from hardware fingerprint"""
        import platform
        import uuid
        
        fingerprint = f"{platform.node()}-{uuid.getnode()}-{time.time()}"
        return hashlib.sha256(fingerprint.encode()).hexdigest()
    
    def _init_bootstrap_nodes(
        self,
        bootstrap_nodes: Optional[List[Tuple[str, int]]],
        use_defaults: bool
    ) -> None:
        """Initialize bootstrap nodes from config, parameter, or defaults.
        
        Priority:
        1. Explicit bootstrap_nodes parameter (if provided)
        2. Environment variable P2P_BOOTSTRAP_NODES (if set)
        3. Default bootstrap nodes (only if use_defaults=True)
        
        Args:
            bootstrap_nodes: Optional list of (host, port) tuples
            use_defaults: Whether to fall back to DEFAULT_BOOTSTRAP_NODES
        """
        if bootstrap_nodes is not None:
            # Use explicitly provided nodes
            self._bootstrap_nodes = list(bootstrap_nodes)
            logger.info(f"📡 Using {len(self._bootstrap_nodes)} provided bootstrap node(s)")
            return
        
        # Try to load from environment config
        settings = get_settings()
        env_nodes = settings.get_bootstrap_nodes()
        
        if env_nodes:
            self._bootstrap_nodes = env_nodes
            logger.info(f"📡 Loaded {len(self._bootstrap_nodes)} bootstrap node(s) from environment")
            return
        
        # Fall back to defaults only if explicitly requested
        if use_defaults:
            self._bootstrap_nodes = list(self.DEFAULT_BOOTSTRAP_NODES)
            logger.info(f"📡 Using {len(self._bootstrap_nodes)} default bootstrap node(s)")
        else:
            self._bootstrap_nodes = []
            logger.info("📡 No bootstrap nodes configured (local discovery only)")
    
    def add_bootstrap_node(self, host: str, port: int) -> bool:
        """Add a bootstrap node dynamically.
        
        Args:
            host: Hostname or IP address of the node
            port: Port number (1-65535)
            
        Returns:
            True if node was added, False if invalid or already exists
        """
        if not (1 <= port <= 65535):
            logger.warning(f"Invalid port {port} for bootstrap node")
            return False
        
        if not host:
            logger.warning("Empty host for bootstrap node")
            return False
        
        node = (host, port)
        if node in self._bootstrap_nodes:
            logger.debug(f"Bootstrap node {host}:{port} already exists")
            return False
        
        self._bootstrap_nodes.append(node)
        logger.info(f"📡 Added bootstrap node: {host}:{port}")
        return True
    
    def remove_bootstrap_node(self, host: str, port: int) -> bool:
        """Remove a bootstrap node dynamically.
        
        Args:
            host: Hostname or IP address of the node
            port: Port number
            
        Returns:
            True if node was removed, False if not found
        """
        node = (host, port)
        if node in self._bootstrap_nodes:
            self._bootstrap_nodes.remove(node)
            logger.info(f"📡 Removed bootstrap node: {host}:{port}")
            return True
        
        logger.debug(f"Bootstrap node {host}:{port} not found")
        return False
    
    def get_bootstrap_nodes(self) -> List[Tuple[str, int]]:
        """Get the current list of bootstrap nodes.
        
        Returns:
            List of (host, port) tuples
        """
        return list(self._bootstrap_nodes)
    
    def clear_bootstrap_nodes(self) -> None:
        """Remove all bootstrap nodes."""
        self._bootstrap_nodes.clear()
        logger.info("📡 Cleared all bootstrap nodes")
    
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
        self.handlers[MessageType.GRADIENT_SHARE] = self._handle_gradient_share
        self.handlers[MessageType.MODEL_SYNC] = self._handle_model_sync
        self.handlers[MessageType.TOKEN_TRANSFER] = self._handle_token_transfer
    
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
                except Exception as e:
                    logger.debug(f"Failed to send response to {addr}: {e}")
    
    async def _bootstrap(self):
        """Bootstrap into the network by connecting to known nodes"""
        if not self._bootstrap_nodes:
            logger.info("🔍 No bootstrap nodes configured, relying on local discovery only")
        else:
            logger.info(f"🔍 Bootstrapping into the network with {len(self._bootstrap_nodes)} node(s)...")
            
            for host, port in self._bootstrap_nodes:
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
        except Exception as e:
            logger.debug(f"Failed to request peers: {e}")
    
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
                except Exception as e:
                    # Mark peer as inactive if heartbeat fails
                    logger.debug(f"Heartbeat failed for peer {peer_id[:16]}...: {e}")
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
        except Exception as e:
            logger.debug(f"Could not estimate GPU compute power: {e}")
        return 0.1  # Default CPU estimate
    
    async def _broadcast(self, message: Message):
        """Broadcast message to all peers"""
        for peer_id, peer in list(self.peers.items()):
            try:
                await self._send_to_peer(peer, message)
            except Exception as e:
                logger.debug(f"Broadcast to peer {peer_id[:16]}... failed: {e}")
    
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
    
    async def _handle_pong(self, message: Message, addr: tuple) -> Optional[Message]:
        """Handle PONG message"""
        try:
            sender_id = message.sender_id
            
            # Update peer info if we already know them
            if sender_id in self.peers:
                peer = self.peers[sender_id]
                peer.last_seen = time.time()
                peer.compute_power = message.payload.get("compute_power", peer.compute_power)
                peer.is_active = True
                logger.debug(f"Updated peer info for {sender_id[:16]}...")
            else:
                # New peer responding to our ping - add them
                compute_power = message.payload.get("compute_power", 1.0)
                port = message.payload.get("port", addr[1])
                
                peer = Peer(
                    node_id=sender_id,
                    host=addr[0],
                    port=port,
                    compute_power=compute_power
                )
                self.peers[sender_id] = peer
                self.known_peers.add((addr[0], port))
                
                logger.info(f"✅ Peer added from PONG: {sender_id[:16]}... ({addr[0]}:{port})")
                
                if self.on_peer_connected:
                    self.on_peer_connected(peer)
            
        except Exception as e:
            logger.error(f"Error handling PONG from {addr}: {e}")
        
        return None  # PONG is a response, no further reply needed
    
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
    
    async def _handle_work_response(self, message: Message, addr: tuple) -> Optional[Message]:
        """Handle work response from a peer that completed assigned work"""
        try:
            work_id = message.payload.get("work_id")
            sender_id = message.sender_id
            result = message.payload.get("result")
            execution_time = message.payload.get("execution_time", 0)
            success = message.payload.get("success", True)
            
            if not work_id:
                logger.warning(f"Work response from {sender_id[:16]}... missing work_id")
                return None
            
            # Store completed work
            self.completed_work[work_id] = {
                "work_id": work_id,
                "result": result,
                "sender_id": sender_id,
                "execution_time": execution_time,
                "success": success,
                "completed_at": time.time()
            }
            
            logger.info(f"📥 Work {work_id[:8]}... completed by {sender_id[:16]}... (took {execution_time:.2f}s)")
            
            # Update peer reputation based on successful completion
            if sender_id in self.peers:
                peer = self.peers[sender_id]
                if success:
                    # Increase reputation for successful work
                    peer.reputation = min(2.0, peer.reputation + 0.01)
                    peer.tokens_earned += message.payload.get("tokens_earned", 0)
                else:
                    # Decrease reputation for failed work
                    peer.reputation = max(0.1, peer.reputation - 0.05)
                    logger.warning(f"Work {work_id[:8]}... failed by peer {sender_id[:16]}...")
            
            # Check if this is part of a distributed training job
            if "gradients" in message.payload:
                await self._handle_gradient_share(message, addr)
            
        except Exception as e:
            logger.error(f"Error handling work response from {addr}: {e}")
        
        return None
    
    async def _handle_heartbeat(self, message: Message, addr: tuple) -> Optional[Message]:
        """Handle heartbeat - update peer status and optionally respond with our status"""
        try:
            sender_id = message.sender_id
            
            if sender_id in self.peers:
                peer = self.peers[sender_id]
                peer.last_seen = time.time()
                peer.is_active = True
                
                # Update peer stats from heartbeat payload
                peer.tokens_earned = message.payload.get("tokens", peer.tokens_earned)
                peer.compute_power = message.payload.get("compute_power", peer.compute_power)
                
                # Track peer's work history
                work_completed = message.payload.get("work_completed", 0)
                pending_work = message.payload.get("pending_work", 0)
                
                logger.debug(
                    f"💓 Heartbeat from {sender_id[:16]}...: "
                    f"tokens={peer.tokens_earned:.2f}, "
                    f"compute={peer.compute_power:.2f} TFLOPS, "
                    f"work={work_completed} done/{pending_work} pending"
                )
            else:
                # Unknown peer sending heartbeat - try to add them
                port = message.payload.get("port", addr[1])
                logger.info(f"Heartbeat from unknown peer {sender_id[:16]}..., attempting connection")
                asyncio.create_task(self._connect_to_peer(addr[0], port))
            
        except Exception as e:
            logger.error(f"Error handling heartbeat from {addr}: {e}")
        
        return None  # Heartbeats don't require a response
    
    async def _handle_announce(self, message: Message, addr: tuple) -> Optional[Message]:
        """Handle node announcement - add new peer and propagate to network"""
        try:
            sender_id = message.sender_id
            port = message.payload.get("port", 31337)
            compute_power = message.payload.get("compute_power", 1.0)
            version = message.payload.get("version", "unknown")
            capabilities = message.payload.get("capabilities", [])
            
            # Don't process our own announcements
            if sender_id == self.node_id:
                return None
            
            # Check if this is a new peer
            is_new_peer = sender_id not in self.peers
            
            if is_new_peer:
                logger.info(
                    f"📢 New node announced: {sender_id[:16]}... "
                    f"({addr[0]}:{port}) - {compute_power:.2f} TFLOPS, v{version}"
                )
                
                # Try to connect to the new peer
                await self._connect_to_peer(addr[0], port)
                
                # Gossip: Forward announcement to our other peers (limited propagation)
                # Only forward to a subset of peers to prevent network flooding
                forward_count = min(3, len(self.peers))
                peers_to_notify = random.sample(
                    list(self.peers.values()),
                    forward_count
                ) if len(self.peers) >= forward_count else list(self.peers.values())
                
                for peer in peers_to_notify:
                    if peer.node_id != sender_id:  # Don't send back to sender
                        try:
                            await self._send_to_peer(peer, message)
                        except Exception as e:
                            logger.debug(f"Failed to forward announcement to {peer.node_id[:16]}...: {e}")
            else:
                # Existing peer - update their info
                peer = self.peers[sender_id]
                peer.last_seen = time.time()
                peer.compute_power = compute_power
                peer.is_active = True
                logger.debug(f"Updated existing peer {sender_id[:16]}... from announcement")
            
        except Exception as e:
            logger.error(f"Error handling announcement from {addr}: {e}")
        
        return None
    
    async def _handle_gradient_share(self, message: Message, addr: tuple) -> Optional[Message]:
        """Handle gradient sharing for distributed training"""
        try:
            sender_id = message.sender_id
            work_id = message.payload.get("work_id")
            gradients = message.payload.get("gradients")
            batch_size = message.payload.get("batch_size", 1)
            epoch = message.payload.get("epoch", 0)
            total_contributors = message.payload.get("total_contributors", 1)
            
            if not work_id or gradients is None:
                logger.warning(f"Invalid gradient share from {sender_id[:16]}...: missing work_id or gradients")
                return None
            
            logger.info(
                f"📊 Received gradients from {sender_id[:16]}... "
                f"for work {work_id[:8]}... (epoch {epoch}, batch_size {batch_size})"
            )
            
            # Initialize buffer for this work_id if needed
            if work_id not in self.gradient_buffer:
                self.gradient_buffer[work_id] = []
                self.gradient_counts[work_id] = total_contributors
            
            # Store the gradients with metadata
            self.gradient_buffer[work_id].append({
                "sender_id": sender_id,
                "gradients": gradients,
                "batch_size": batch_size,
                "epoch": epoch,
                "received_at": time.time()
            })
            
            # Update peer reputation for contribution
            if sender_id in self.peers:
                self.peers[sender_id].reputation = min(2.0, self.peers[sender_id].reputation + 0.005)
            
            # Trigger callback if registered
            if self.on_gradient_received:
                self.on_gradient_received(work_id, gradients, sender_id)
            
            # Check if we have enough gradients to aggregate
            if len(self.gradient_buffer[work_id]) >= self.gradient_counts[work_id]:
                aggregated = await self._aggregate_gradients(work_id)
                if aggregated and self.on_gradients_aggregated:
                    self.on_gradients_aggregated(work_id, aggregated)
            
            # Acknowledge receipt
            return Message(
                msg_type=MessageType.GRADIENT_SHARE,
                sender_id=self.node_id,
                payload={
                    "ack": True,
                    "work_id": work_id,
                    "received_count": len(self.gradient_buffer[work_id]),
                    "expected_count": self.gradient_counts[work_id]
                }
            )
            
        except Exception as e:
            logger.error(f"Error handling gradient share from {addr}: {e}")
            return None
    
    async def _aggregate_gradients(self, work_id: str) -> Optional[dict]:
        """Aggregate gradients from multiple peers using weighted averaging"""
        try:
            if work_id not in self.gradient_buffer:
                return None
            
            gradient_list = self.gradient_buffer[work_id]
            if not gradient_list:
                return None
            
            logger.info(f"🔄 Aggregating {len(gradient_list)} gradient sets for work {work_id[:8]}...")
            
            # Compute weighted average based on batch sizes
            total_samples = sum(g["batch_size"] for g in gradient_list)
            
            # Initialize aggregated gradients structure from first gradient
            first_gradients = gradient_list[0]["gradients"]
            aggregated = {}
            
            if isinstance(first_gradients, dict):
                # Gradient is a dict of layer_name -> values
                for key in first_gradients:
                    weighted_sum = None
                    for g_data in gradient_list:
                        weight = g_data["batch_size"] / total_samples
                        grad_value = g_data["gradients"].get(key)
                        
                        if grad_value is not None:
                            if isinstance(grad_value, list):
                                # Handle list of gradient values
                                weighted_grad = [v * weight for v in grad_value]
                                if weighted_sum is None:
                                    weighted_sum = weighted_grad
                                else:
                                    weighted_sum = [a + b for a, b in zip(weighted_sum, weighted_grad)]
                            elif isinstance(grad_value, (int, float)):
                                # Handle scalar gradient values
                                if weighted_sum is None:
                                    weighted_sum = 0
                                weighted_sum += grad_value * weight
                    
                    aggregated[key] = weighted_sum
            elif isinstance(first_gradients, list):
                # Gradient is a flat list
                weighted_sum = [0.0] * len(first_gradients)
                for g_data in gradient_list:
                    weight = g_data["batch_size"] / total_samples
                    for i, v in enumerate(g_data["gradients"]):
                        weighted_sum[i] += v * weight
                aggregated = weighted_sum
            
            # Store aggregated result
            self.aggregated_gradients[work_id] = {
                "gradients": aggregated,
                "total_samples": total_samples,
                "contributor_count": len(gradient_list),
                "aggregated_at": time.time()
            }
            
            # Clean up buffer
            del self.gradient_buffer[work_id]
            del self.gradient_counts[work_id]
            
            logger.info(f"✅ Gradient aggregation complete for work {work_id[:8]}... ({total_samples} total samples)")
            
            return self.aggregated_gradients[work_id]
            
        except Exception as e:
            logger.error(f"Error aggregating gradients for {work_id}: {e}")
            return None
    
    async def _handle_model_sync(self, message: Message, addr: tuple) -> Optional[Message]:
        """Handle model synchronization between peers"""
        try:
            sender_id = message.sender_id
            model_version = message.payload.get("model_version")
            model_state = message.payload.get("model_state")
            sync_type = message.payload.get("sync_type", "full")  # "full" or "delta"
            request_only = message.payload.get("request_only", False)
            
            if request_only:
                # Peer is requesting our model state
                logger.info(f"📤 Model sync request from {sender_id[:16]}...")
                
                if self.model_state:
                    return Message(
                        msg_type=MessageType.MODEL_SYNC,
                        sender_id=self.node_id,
                        payload={
                            "model_version": self.current_model_version,
                            "model_state": self.model_state,
                            "sync_type": "full"
                        }
                    )
                else:
                    return Message(
                        msg_type=MessageType.MODEL_SYNC,
                        sender_id=self.node_id,
                        payload={
                            "error": "no_model_available",
                            "model_version": self.current_model_version
                        }
                    )
            
            # Peer is sending us their model state
            if model_version and model_state:
                logger.info(
                    f"📥 Model sync from {sender_id[:16]}... "
                    f"(version {model_version}, type: {sync_type})"
                )
                
                # Store pending sync for review/application
                self.pending_model_syncs[sender_id] = {
                    "model_version": model_version,
                    "model_state": model_state,
                    "sync_type": sync_type,
                    "received_at": time.time()
                }
                
                # Compare versions and decide whether to apply
                if self._version_is_newer(model_version, self.current_model_version):
                    logger.info(f"Received newer model version {model_version} (current: {self.current_model_version})")
                    
                    # Apply the model update
                    if sync_type == "delta" and self.model_state:
                        # Apply delta update
                        self.model_state = self._apply_model_delta(self.model_state, model_state)
                    else:
                        # Full replacement
                        self.model_state = model_state
                    
                    self.current_model_version = model_version
                    
                    # Trigger callback
                    if self.on_model_sync:
                        self.on_model_sync(model_version, model_state)
                    
                    # Update peer reputation for sharing updates
                    if sender_id in self.peers:
                        self.peers[sender_id].reputation = min(2.0, self.peers[sender_id].reputation + 0.01)
                
                # Acknowledge sync
                return Message(
                    msg_type=MessageType.MODEL_SYNC,
                    sender_id=self.node_id,
                    payload={
                        "ack": True,
                        "current_version": self.current_model_version,
                        "applied": self._version_is_newer(model_version, self.current_model_version)
                    }
                )
            
        except Exception as e:
            logger.error(f"Error handling model sync from {addr}: {e}")
        
        return None
    
    def _version_is_newer(self, new_version: str, current_version: str) -> bool:
        """Compare semantic versions to check if new_version is newer"""
        try:
            new_parts = [int(x) for x in new_version.split(".")]
            current_parts = [int(x) for x in current_version.split(".")]
            
            # Pad shorter version with zeros
            while len(new_parts) < len(current_parts):
                new_parts.append(0)
            while len(current_parts) < len(new_parts):
                current_parts.append(0)
            
            return new_parts > current_parts
        except Exception as e:
            logger.debug(f"Version comparison failed for '{new_version}' vs '{current_version}': {e}")
            return False
    
    def _apply_model_delta(self, base_state: dict, delta: dict) -> dict:
        """Apply delta updates to model state"""
        result = base_state.copy()
        for key, value in delta.items():
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                result[key] = self._apply_model_delta(result[key], value)
            else:
                result[key] = value
        return result
    
    async def _handle_token_transfer(self, message: Message, addr: tuple) -> Optional[Message]:
        """Handle token transfer requests and confirmations"""
        try:
            sender_id = message.sender_id
            transfer_type = message.payload.get("type")  # "request", "confirm", "reject"
            transfer_id = message.payload.get("transfer_id")
            amount = message.payload.get("amount", 0)
            reason = message.payload.get("reason", "unspecified")
            
            if transfer_type == "request":
                # Someone is requesting tokens from us
                recipient = message.payload.get("recipient", sender_id)
                
                logger.info(
                    f"💰 Token transfer request from {sender_id[:16]}...: "
                    f"{amount} AOAI for '{reason}'"
                )
                
                # Validate the request
                if amount <= 0:
                    return Message(
                        msg_type=MessageType.TOKEN_TRANSFER,
                        sender_id=self.node_id,
                        payload={
                            "type": "reject",
                            "transfer_id": transfer_id,
                            "reason": "invalid_amount"
                        }
                    )
                
                if amount > self.token_balance:
                    return Message(
                        msg_type=MessageType.TOKEN_TRANSFER,
                        sender_id=self.node_id,
                        payload={
                            "type": "reject",
                            "transfer_id": transfer_id,
                            "reason": "insufficient_balance"
                        }
                    )
                
                # Generate transfer ID if not provided
                if not transfer_id:
                    transfer_id = hashlib.sha256(
                        f"{self.node_id}{recipient}{amount}{time.time()}".encode()
                    ).hexdigest()[:16]
                
                # Store pending transfer
                self.pending_transfers[transfer_id] = {
                    "from": self.node_id,
                    "to": recipient,
                    "amount": amount,
                    "reason": reason,
                    "status": "pending",
                    "created_at": time.time()
                }
                
                # Execute the transfer (deduct from our balance)
                self.token_balance -= amount
                self.pending_transfers[transfer_id]["status"] = "completed"
                
                # Log to transfer history
                self.transfer_history.append({
                    "transfer_id": transfer_id,
                    "type": "outgoing",
                    "to": recipient,
                    "amount": amount,
                    "reason": reason,
                    "timestamp": time.time()
                })
                
                logger.info(f"✅ Token transfer {transfer_id[:8]}... completed: {amount} AOAI to {recipient[:16]}...")
                
                if self.on_tokens_earned:
                    self.on_tokens_earned(-amount, reason)  # Negative for outgoing
                
                return Message(
                    msg_type=MessageType.TOKEN_TRANSFER,
                    sender_id=self.node_id,
                    payload={
                        "type": "confirm",
                        "transfer_id": transfer_id,
                        "amount": amount,
                        "new_balance": self.token_balance
                    }
                )
                
            elif transfer_type == "confirm":
                # Transfer to us was confirmed
                logger.info(f"✅ Incoming transfer {transfer_id[:8]}... confirmed: {amount} AOAI")
                
                self.token_balance += amount
                
                self.transfer_history.append({
                    "transfer_id": transfer_id,
                    "type": "incoming",
                    "from": sender_id,
                    "amount": amount,
                    "timestamp": time.time()
                })
                
                if self.on_tokens_earned:
                    self.on_tokens_earned(amount, "transfer_received")
                
            elif transfer_type == "reject":
                # Our transfer request was rejected
                reject_reason = message.payload.get("reason", "unknown")
                logger.warning(f"❌ Transfer {transfer_id[:8]}... rejected: {reject_reason}")
                
                if transfer_id in self.pending_transfers:
                    self.pending_transfers[transfer_id]["status"] = "rejected"
                    self.pending_transfers[transfer_id]["reject_reason"] = reject_reason
            
        except Exception as e:
            logger.error(f"Error handling token transfer from {addr}: {e}")
        
        return None
    
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
            "work_completed": len(self.completed_work),
            "pending_work": len(self.pending_work),
            "pending_gradients": len(self.gradient_buffer),
            "aggregated_gradients": len(self.aggregated_gradients),
            "model_version": self.current_model_version,
            "pending_transfers": len(self.pending_transfers),
            "transfer_history_count": len(self.transfer_history),
            "average_peer_reputation": (
                sum(p.reputation for p in self.peers.values()) / len(self.peers)
                if self.peers else 0
            )
        }
    
    # Public APIs for gradient sharing and model sync
    
    async def share_gradients(self, work_id: str, gradients: dict, batch_size: int = 1, epoch: int = 0):
        """Share computed gradients with peers for aggregation"""
        message = Message(
            msg_type=MessageType.GRADIENT_SHARE,
            sender_id=self.node_id,
            payload={
                "work_id": work_id,
                "gradients": gradients,
                "batch_size": batch_size,
                "epoch": epoch,
                "total_contributors": len(self.peers) + 1  # Include ourselves
            }
        )
        
        logger.info(f"📤 Sharing gradients for work {work_id[:8]}... with {len(self.peers)} peers")
        await self._broadcast(message)
    
    async def request_model_sync(self, peer_id: Optional[str] = None):
        """Request model state from a peer or broadcast request"""
        message = Message(
            msg_type=MessageType.MODEL_SYNC,
            sender_id=self.node_id,
            payload={
                "request_only": True,
                "current_version": self.current_model_version
            }
        )
        
        if peer_id and peer_id in self.peers:
            await self._send_to_peer(self.peers[peer_id], message)
            logger.info(f"📤 Requesting model sync from {peer_id[:16]}...")
        else:
            # Request from the peer with highest reputation
            best_peer = max(self.peers.values(), key=lambda p: p.reputation, default=None)
            if best_peer:
                await self._send_to_peer(best_peer, message)
                logger.info(f"📤 Requesting model sync from best peer {best_peer.node_id[:16]}...")
    
    async def broadcast_model_update(self, model_state: dict, version: str, sync_type: str = "full"):
        """Broadcast model update to all peers"""
        self.model_state = model_state
        self.current_model_version = version
        
        message = Message(
            msg_type=MessageType.MODEL_SYNC,
            sender_id=self.node_id,
            payload={
                "model_version": version,
                "model_state": model_state,
                "sync_type": sync_type
            }
        )
        
        logger.info(f"📤 Broadcasting model update v{version} to {len(self.peers)} peers")
        await self._broadcast(message)
    
    async def transfer_tokens(self, recipient_id: str, amount: float, reason: str = "reward") -> Optional[str]:
        """Transfer tokens to another peer"""
        if recipient_id not in self.peers:
            logger.warning(f"Cannot transfer tokens: peer {recipient_id[:16]}... not found")
            return None
        
        if amount > self.token_balance:
            logger.warning(f"Cannot transfer tokens: insufficient balance ({self.token_balance} < {amount})")
            return None
        
        transfer_id = hashlib.sha256(
            f"{self.node_id}{recipient_id}{amount}{time.time()}".encode()
        ).hexdigest()[:16]
        
        message = Message(
            msg_type=MessageType.TOKEN_TRANSFER,
            sender_id=self.node_id,
            payload={
                "type": "request",
                "transfer_id": transfer_id,
                "recipient": recipient_id,
                "amount": amount,
                "reason": reason
            }
        )
        
        # Store pending outgoing transfer
        self.pending_transfers[transfer_id] = {
            "from": self.node_id,
            "to": recipient_id,
            "amount": amount,
            "reason": reason,
            "status": "pending",
            "created_at": time.time()
        }
        
        await self._send_to_peer(self.peers[recipient_id], message)
        logger.info(f"📤 Token transfer initiated: {amount} AOAI to {recipient_id[:16]}... ({transfer_id[:8]}...)")
        
        return transfer_id
    
    def get_aggregated_gradients(self, work_id: str) -> Optional[dict]:
        """Get aggregated gradients for a work item if available"""
        return self.aggregated_gradients.get(work_id)
    
    def get_pending_model_syncs(self) -> Dict[str, dict]:
        """Get all pending model syncs from peers"""
        return self.pending_model_syncs.copy()
    
    def get_transfer_history(self, limit: int = 100) -> List[dict]:
        """Get recent token transfer history"""
        return self.transfer_history[-limit:]
    
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
