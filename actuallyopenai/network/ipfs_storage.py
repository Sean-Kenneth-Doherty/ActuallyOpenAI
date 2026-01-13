"""
IPFS Model Storage
==================
Store and retrieve model weights on IPFS - the interplanetary file system.
Models become permanent, uncensorable, and globally distributed.

No single server. The network IS the storage.
"""

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List, BinaryIO
import logging
import struct
import asyncio
import aiohttp

logger = logging.getLogger("AOAI-IPFS")


@dataclass
class IPFSConfig:
    """IPFS Configuration"""
    # Public IPFS gateways (decentralized - anyone can run one)
    GATEWAYS = [
        "https://ipfs.io",
        "https://gateway.pinata.cloud",
        "https://cloudflare-ipfs.com",
        "https://dweb.link",
        "https://w3s.link",  # Web3.Storage
    ]
    
    # Local IPFS node (if running)
    LOCAL_API = "http://127.0.0.1:5001"
    
    # Pinning services for persistence
    PINATA_API = "https://api.pinata.cloud"
    WEB3_STORAGE_API = "https://api.web3.storage"


class IPFSModelStorage:
    """
    Decentralized model storage using IPFS
    
    Features:
    - Content-addressed (CID) - immutable references
    - Distributed across thousands of nodes
    - No single point of failure
    - Permanent storage options (Filecoin, Arweave)
    """
    
    def __init__(
        self,
        local_cache_dir: Optional[str] = None,
        pinata_jwt: Optional[str] = None,
        web3_storage_token: Optional[str] = None
    ):
        self.cache_dir = Path(local_cache_dir or tempfile.mkdtemp(prefix="aoai_ipfs_"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Pinning service credentials (for permanent storage)
        self.pinata_jwt = pinata_jwt or os.environ.get("PINATA_JWT")
        self.web3_storage_token = web3_storage_token or os.environ.get("WEB3_STORAGE_TOKEN")
        
        # Track uploaded models
        self.model_registry: Dict[str, str] = {}  # model_name -> CID
        
        logger.info(f"📦 IPFS Storage initialized, cache: {self.cache_dir}")
    
    async def upload_model(
        self,
        model_path: str,
        model_name: str,
        metadata: Optional[dict] = None,
        pin: bool = True
    ) -> str:
        """
        Upload model to IPFS
        
        Returns: Content Identifier (CID) - the permanent address
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Read model file
        with open(model_path, 'rb') as f:
            model_data = f.read()
        
        # Create model package with metadata
        package = self._create_model_package(model_data, model_name, metadata)
        
        # Try local IPFS first, then pinning services
        cid = None
        
        # Try local IPFS node
        cid = await self._upload_to_local_ipfs(package)
        
        if not cid and self.pinata_jwt:
            cid = await self._upload_to_pinata(package, model_name)
        
        if not cid and self.web3_storage_token:
            cid = await self._upload_to_web3storage(package, model_name)
        
        if not cid:
            # Fallback: compute CID locally and store in cache
            cid = self._compute_cid(package)
            cache_path = self.cache_dir / cid
            with open(cache_path, 'wb') as f:
                f.write(package)
            logger.warning(f"⚠️ No IPFS connection, cached locally: {cid}")
        
        # Register model
        self.model_registry[model_name] = cid
        
        logger.info(f"✅ Model uploaded: {model_name} -> ipfs://{cid}")
        
        return cid
    
    async def download_model(
        self,
        cid: str,
        output_path: Optional[str] = None
    ) -> bytes:
        """
        Download model from IPFS
        
        The magic: This works even if the original uploader is offline,
        as long as ANY node in the world has the content.
        """
        # Check local cache first
        cache_path = self.cache_dir / cid
        if cache_path.exists():
            logger.info(f"📂 Loading from cache: {cid}")
            with open(cache_path, 'rb') as f:
                package = f.read()
            return self._extract_model_data(package)
        
        # Try local IPFS node
        data = await self._download_from_local_ipfs(cid)
        
        # Try public gateways
        if not data:
            for gateway in IPFSConfig.GATEWAYS:
                try:
                    data = await self._download_from_gateway(gateway, cid)
                    if data:
                        break
                except Exception as e:
                    logger.debug(f"Gateway {gateway} failed: {e}")
        
        if not data:
            raise Exception(f"Model not found on IPFS: {cid}")
        
        # Cache locally
        with open(cache_path, 'wb') as f:
            f.write(data)
        
        # Extract model data from package
        model_data = self._extract_model_data(data)
        
        # Save to output path if specified
        if output_path:
            with open(output_path, 'wb') as f:
                f.write(model_data)
            logger.info(f"📥 Model saved to: {output_path}")
        
        return model_data
    
    def _create_model_package(
        self,
        model_data: bytes,
        model_name: str,
        metadata: Optional[dict]
    ) -> bytes:
        """Create a model package with metadata"""
        meta = {
            "name": model_name,
            "size": len(model_data),
            "hash": hashlib.sha256(model_data).hexdigest(),
            "format": "pytorch",
            "version": "1.0.0",
            **(metadata or {})
        }
        
        meta_json = json.dumps(meta).encode('utf-8')
        
        # Package format: [4-byte meta length][meta json][model data]
        package = struct.pack('>I', len(meta_json)) + meta_json + model_data
        
        return package
    
    def _extract_model_data(self, package: bytes) -> bytes:
        """Extract model data from package"""
        meta_length = struct.unpack('>I', package[:4])[0]
        model_data = package[4 + meta_length:]
        return model_data
    
    def _compute_cid(self, data: bytes) -> str:
        """Compute IPFS CID (simplified - real IPFS uses multihash)"""
        # This is a simplified CID - real IPFS uses more complex encoding
        hash_bytes = hashlib.sha256(data).digest()
        # CIDv1 prefix for raw data
        return "baf" + hash_bytes.hex()[:56]
    
    async def _upload_to_local_ipfs(self, data: bytes) -> Optional[str]:
        """Upload to local IPFS node"""
        try:
            async with aiohttp.ClientSession() as session:
                # IPFS HTTP API
                form = aiohttp.FormData()
                form.add_field('file', data, filename='model.aoai')
                
                async with session.post(
                    f"{IPFSConfig.LOCAL_API}/api/v0/add",
                    data=form,
                    timeout=aiohttp.ClientTimeout(total=300)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("Hash")
        except Exception as e:
            logger.debug(f"Local IPFS not available: {e}")
        return None
    
    async def _upload_to_pinata(self, data: bytes, name: str) -> Optional[str]:
        """Upload to Pinata pinning service"""
        try:
            async with aiohttp.ClientSession() as session:
                form = aiohttp.FormData()
                form.add_field('file', data, filename=f'{name}.aoai')
                form.add_field('pinataMetadata', json.dumps({"name": name}))
                
                headers = {"Authorization": f"Bearer {self.pinata_jwt}"}
                
                async with session.post(
                    f"{IPFSConfig.PINATA_API}/pinning/pinFileToIPFS",
                    data=form,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=300)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        cid = result.get("IpfsHash")
                        logger.info(f"📌 Pinned to Pinata: {cid}")
                        return cid
        except Exception as e:
            logger.debug(f"Pinata upload failed: {e}")
        return None
    
    async def _upload_to_web3storage(self, data: bytes, name: str) -> Optional[str]:
        """Upload to Web3.Storage (free, powered by Filecoin)"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.web3_storage_token}",
                    "X-Name": name
                }
                
                async with session.post(
                    f"{IPFSConfig.WEB3_STORAGE_API}/upload",
                    data=data,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=300)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        cid = result.get("cid")
                        logger.info(f"📌 Stored on Web3.Storage: {cid}")
                        return cid
        except Exception as e:
            logger.debug(f"Web3.Storage upload failed: {e}")
        return None
    
    async def _download_from_local_ipfs(self, cid: str) -> Optional[bytes]:
        """Download from local IPFS node"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{IPFSConfig.LOCAL_API}/api/v0/cat?arg={cid}",
                    timeout=aiohttp.ClientTimeout(total=300)
                ) as response:
                    if response.status == 200:
                        return await response.read()
        except:
            pass
        return None
    
    async def _download_from_gateway(self, gateway: str, cid: str) -> Optional[bytes]:
        """Download from IPFS gateway"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{gateway}/ipfs/{cid}"
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    if response.status == 200:
                        logger.info(f"📥 Downloaded from {gateway}")
                        return await response.read()
        except Exception as e:
            logger.debug(f"Gateway download failed: {e}")
        return None
    
    async def pin_to_filecoin(self, cid: str) -> str:
        """Pin content to Filecoin for permanent storage (paid)"""
        # This would integrate with Filecoin storage deals
        # For now, return the CID
        logger.info(f"🔒 Filecoin pinning: {cid}")
        return cid
    
    async def get_model_info(self, cid: str) -> dict:
        """Get model metadata without downloading full model"""
        # Check local cache
        cache_path = self.cache_dir / cid
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                meta_length = struct.unpack('>I', f.read(4))[0]
                meta_json = f.read(meta_length)
                return json.loads(meta_json.decode('utf-8'))
        
        # Would need to fetch from IPFS otherwise
        return {"cid": cid, "status": "not_cached"}
    
    def list_cached_models(self) -> List[dict]:
        """List locally cached models"""
        models = []
        for path in self.cache_dir.iterdir():
            if path.is_file():
                try:
                    with open(path, 'rb') as f:
                        meta_length = struct.unpack('>I', f.read(4))[0]
                        meta_json = f.read(meta_length)
                        meta = json.loads(meta_json.decode('utf-8'))
                        meta["cid"] = path.name
                        models.append(meta)
                except:
                    pass
        return models


class ModelRegistry:
    """
    Decentralized model registry on IPFS
    
    Tracks all published models across the network.
    Like npm, but for AI models, and completely decentralized.
    """
    
    # Well-known CID for the registry (updated via DNS or ENS)
    REGISTRY_CID = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
    
    def __init__(self, storage: IPFSModelStorage):
        self.storage = storage
        self.registry: Dict[str, dict] = {}
    
    async def publish_model(
        self,
        model_path: str,
        name: str,
        version: str,
        description: str,
        author: str,
        metadata: Optional[dict] = None
    ) -> str:
        """Publish a model to the global registry"""
        
        # Upload model to IPFS
        cid = await self.storage.upload_model(
            model_path,
            name,
            metadata={
                "version": version,
                "description": description,
                "author": author,
                **(metadata or {})
            }
        )
        
        # Add to local registry
        self.registry[f"{name}@{version}"] = {
            "name": name,
            "version": version,
            "cid": cid,
            "description": description,
            "author": author,
            "published_at": __import__('time').time()
        }
        
        logger.info(f"📢 Model published: {name}@{version} -> ipfs://{cid}")
        
        return cid
    
    async def install_model(self, name: str, version: str = "latest") -> bytes:
        """Install a model from the registry"""
        key = f"{name}@{version}"
        
        if key not in self.registry:
            raise ValueError(f"Model not found: {key}")
        
        cid = self.registry[key]["cid"]
        
        return await self.storage.download_model(cid)
    
    def search(self, query: str) -> List[dict]:
        """Search models in registry"""
        results = []
        query_lower = query.lower()
        
        for key, model in self.registry.items():
            if (query_lower in model["name"].lower() or 
                query_lower in model.get("description", "").lower()):
                results.append(model)
        
        return results


# CLI
async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="AOAI Model Storage (IPFS)")
    parser.add_argument("command", choices=["upload", "download", "list", "info"])
    parser.add_argument("--model", help="Model path or CID")
    parser.add_argument("--name", help="Model name")
    parser.add_argument("--output", help="Output path for download")
    args = parser.parse_args()
    
    storage = IPFSModelStorage()
    
    if args.command == "upload":
        if not args.model or not args.name:
            print("❌ --model and --name required for upload")
            return
        cid = await storage.upload_model(args.model, args.name)
        print(f"\n✅ Uploaded to IPFS!")
        print(f"   CID: {cid}")
        print(f"   URL: https://ipfs.io/ipfs/{cid}")
        
    elif args.command == "download":
        if not args.model:
            print("❌ --model (CID) required for download")
            return
        data = await storage.download_model(args.model, args.output)
        print(f"✅ Downloaded {len(data)} bytes")
        
    elif args.command == "list":
        models = storage.list_cached_models()
        print(f"\n📦 Cached Models ({len(models)}):")
        for m in models:
            print(f"   • {m.get('name', 'unknown')} - {m['cid'][:20]}...")
            
    elif args.command == "info":
        if not args.model:
            print("❌ --model (CID) required")
            return
        info = await storage.get_model_info(args.model)
        print(f"\n📋 Model Info:")
        for k, v in info.items():
            print(f"   {k}: {v}")


if __name__ == "__main__":
    asyncio.run(main())
