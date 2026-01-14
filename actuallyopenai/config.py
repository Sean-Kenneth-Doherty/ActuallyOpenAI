"""
Configuration management for OpenCollective AI.
"""

from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Server
    orchestrator_host: str = Field(default="0.0.0.0")
    orchestrator_port: int = Field(default=8000)
    orchestrator_workers: int = Field(default=4)
    debug: bool = Field(default=False)
    
    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://user:password@localhost:5432/actuallyopenai"
    )
    redis_url: str = Field(default="redis://localhost:6379/0")
    
    # Blockchain
    blockchain_network: str = Field(default="sepolia")
    blockchain_rpc_url: str = Field(default="")
    token_contract_address: str = Field(default="")
    treasury_private_key: str = Field(default="")
    treasury_address: str = Field(default="")
    min_payout_threshold: float = Field(default=10.0)
    payout_frequency_hours: int = Field(default=24)
    
    # Worker
    worker_wallet_address: str = Field(default="")
    worker_name: str = Field(default="worker-node")
    worker_region: str = Field(default="unknown")
    max_gpu_memory_gb: int = Field(default=8)
    max_cpu_cores: int = Field(default=4)
    max_ram_gb: int = Field(default=16)
    
    # Security
    jwt_secret_key: str = Field(default="change-this-secret-key")
    jwt_algorithm: str = Field(default="HS256")
    jwt_expiration_hours: int = Field(default=24)
    api_key_salt: str = Field(default="random-salt")
    
    # IPFS
    ipfs_host: str = Field(default="127.0.0.1")
    ipfs_port: int = Field(default=5001)
    ipfs_gateway: str = Field(default="https://ipfs.io/ipfs/")
    
    # Training
    default_batch_size: int = Field(default=32)
    gradient_accumulation_steps: int = Field(default=4)
    checkpoint_frequency: int = Field(default=1000)
    
    # Monitoring
    prometheus_port: int = Field(default=9090)
    enable_metrics: bool = Field(default=True)
    log_level: str = Field(default="INFO")
    
    # P2P Network
    p2p_bootstrap_nodes: str = Field(
        default="",
        description="Comma-separated list of bootstrap nodes in host:port format"
    )
    p2p_port: int = Field(default=31337)
    p2p_max_peers: int = Field(default=50)
    
    def get_bootstrap_nodes(self) -> list:
        """Parse and validate bootstrap nodes from environment variable.
        
        Returns:
            List of tuples (host, port) for valid bootstrap nodes.
        """
        if not self.p2p_bootstrap_nodes:
            return []
        
        nodes = []
        for node in self.p2p_bootstrap_nodes.split(","):
            node = node.strip()
            if not node:
                continue
            
            if not self._validate_node_format(node):
                continue
            
            try:
                host, port_str = node.rsplit(":", 1)
                port = int(port_str)
                if 1 <= port <= 65535:
                    nodes.append((host, port))
            except ValueError:
                continue
        
        return nodes
    
    @staticmethod
    def _validate_node_format(node: str) -> bool:
        """Validate that a node string is in host:port format.
        
        Args:
            node: Node address string to validate.
            
        Returns:
            True if valid format, False otherwise.
        """
        if ":" not in node:
            return False
        
        try:
            host, port_str = node.rsplit(":", 1)
            port = int(port_str)
            return bool(host) and 1 <= port <= 65535
        except ValueError:
            return False
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
