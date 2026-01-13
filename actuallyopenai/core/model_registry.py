"""
Model Registry - Decentralized storage and management of trained AI models.
"""

import hashlib
import os
from datetime import datetime
from decimal import Decimal
from typing import Optional, List, Dict, Any
import uuid

import structlog
from pydantic import BaseModel

from actuallyopenai.config import get_settings
from actuallyopenai.core.models import ModelInfo

logger = structlog.get_logger()


class ModelVersion(BaseModel):
    """A specific version of a model."""
    version: str
    ipfs_hash: str
    file_size_bytes: int
    training_steps: int
    loss: Optional[float]
    created_at: datetime
    changelog: str = ""


class ModelRegistry:
    """
    Registry for managing trained AI models.
    Supports IPFS storage for decentralized model hosting.
    """
    
    def __init__(self, ipfs_host: str = None, ipfs_port: int = None):
        settings = get_settings()
        self.ipfs_host = ipfs_host or settings.ipfs_host
        self.ipfs_port = ipfs_port or settings.ipfs_port
        
        # In-memory registry (would be database in production)
        self.models: Dict[str, ModelInfo] = {}
        self.model_versions: Dict[str, List[ModelVersion]] = {}
        
        # IPFS client (lazy initialization)
        self._ipfs_client = None
    
    @property
    def ipfs_client(self):
        """Lazy initialization of IPFS client."""
        if self._ipfs_client is None:
            try:
                import ipfshttpclient
                self._ipfs_client = ipfshttpclient.connect(
                    f"/ip4/{self.ipfs_host}/tcp/{self.ipfs_port}"
                )
            except Exception as e:
                logger.warning("IPFS client not available", error=str(e))
                self._ipfs_client = None
        return self._ipfs_client
    
    async def register_model(
        self,
        name: str,
        model_type: str,
        architecture: str,
        description: str = "",
        parameter_count: int = 0
    ) -> ModelInfo:
        """
        Register a new model in the registry.
        
        Args:
            name: Human-readable model name
            model_type: Type of model (transformer, cnn, etc.)
            architecture: Specific architecture (gpt2, llama, etc.)
            description: Model description
            parameter_count: Number of parameters
            
        Returns:
            The registered ModelInfo
        """
        model = ModelInfo(
            name=name,
            model_type=model_type,
            architecture=architecture,
            description=description,
            parameter_count=parameter_count
        )
        
        self.models[model.id] = model
        self.model_versions[model.id] = []
        
        logger.info(
            "Model registered",
            model_id=model.id,
            name=name,
            architecture=architecture
        )
        
        return model
    
    async def upload_model_weights(
        self,
        model_id: str,
        weights_path: str,
        version: str = None,
        training_steps: int = 0,
        loss: float = None,
        changelog: str = ""
    ) -> Optional[str]:
        """
        Upload model weights to IPFS and register the version.
        
        Args:
            model_id: The model to upload weights for
            weights_path: Local path to the weights file
            version: Version string (auto-generated if not provided)
            training_steps: Number of training steps
            loss: Final loss value
            changelog: Description of changes in this version
            
        Returns:
            IPFS hash if successful, None otherwise
        """
        if model_id not in self.models:
            logger.error("Model not found", model_id=model_id)
            return None
        
        model = self.models[model_id]
        
        # Calculate file size
        file_size = os.path.getsize(weights_path)
        
        # Upload to IPFS
        ipfs_hash = None
        if self.ipfs_client:
            try:
                result = self.ipfs_client.add(weights_path)
                ipfs_hash = result["Hash"]
                logger.info("Weights uploaded to IPFS", ipfs_hash=ipfs_hash)
            except Exception as e:
                logger.error("Failed to upload to IPFS", error=str(e))
                # Fall back to local hash
                with open(weights_path, "rb") as f:
                    ipfs_hash = f"local-{hashlib.sha256(f.read()).hexdigest()[:32]}"
        else:
            # Generate local hash for testing
            with open(weights_path, "rb") as f:
                ipfs_hash = f"local-{hashlib.sha256(f.read()).hexdigest()[:32]}"
        
        # Auto-generate version
        if not version:
            version_count = len(self.model_versions.get(model_id, []))
            version = f"1.0.{version_count}"
        
        # Create version record
        model_version = ModelVersion(
            version=version,
            ipfs_hash=ipfs_hash,
            file_size_bytes=file_size,
            training_steps=training_steps,
            loss=loss,
            created_at=datetime.utcnow(),
            changelog=changelog
        )
        
        # Update model
        model.ipfs_hash = ipfs_hash
        model.file_size_bytes = file_size
        model.training_steps = training_steps
        model.final_loss = loss
        model.version = version
        model.updated_at = datetime.utcnow()
        
        # Add to versions
        if model_id not in self.model_versions:
            self.model_versions[model_id] = []
        self.model_versions[model_id].append(model_version)
        
        logger.info(
            "Model version registered",
            model_id=model_id,
            version=version,
            ipfs_hash=ipfs_hash
        )
        
        return ipfs_hash
    
    async def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """Get model by ID."""
        return self.models.get(model_id)
    
    async def get_model_by_name(self, name: str) -> Optional[ModelInfo]:
        """Get model by name."""
        for model in self.models.values():
            if model.name.lower() == name.lower():
                return model
        return None
    
    async def list_models(
        self,
        model_type: str = None,
        is_public: bool = None
    ) -> List[ModelInfo]:
        """
        List models with optional filtering.
        
        Args:
            model_type: Filter by model type
            is_public: Filter by public status
            
        Returns:
            List of matching models
        """
        models = list(self.models.values())
        
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        
        if is_public is not None:
            models = [m for m in models if m.is_public == is_public]
        
        return models
    
    async def get_model_versions(self, model_id: str) -> List[ModelVersion]:
        """Get all versions of a model."""
        return self.model_versions.get(model_id, [])
    
    async def get_model_download_url(
        self,
        model_id: str,
        version: str = None
    ) -> Optional[str]:
        """
        Get download URL for model weights.
        
        Args:
            model_id: The model ID
            version: Specific version (latest if not provided)
            
        Returns:
            Download URL (IPFS gateway URL)
        """
        settings = get_settings()
        
        if model_id not in self.models:
            return None
        
        model = self.models[model_id]
        
        if version:
            versions = self.model_versions.get(model_id, [])
            for v in versions:
                if v.version == version:
                    return f"{settings.ipfs_gateway}{v.ipfs_hash}"
            return None
        
        if model.ipfs_hash:
            return f"{settings.ipfs_gateway}{model.ipfs_hash}"
        
        return None
    
    async def record_api_call(self, model_id: str, revenue: Decimal = Decimal("0")):
        """Record an API call for a model."""
        if model_id in self.models:
            self.models[model_id].api_calls += 1
            self.models[model_id].total_revenue += revenue
    
    async def get_model_stats(self, model_id: str) -> Dict[str, Any]:
        """Get statistics for a model."""
        if model_id not in self.models:
            return {}
        
        model = self.models[model_id]
        versions = self.model_versions.get(model_id, [])
        
        return {
            "model_id": model_id,
            "name": model.name,
            "total_versions": len(versions),
            "latest_version": model.version,
            "api_calls": model.api_calls,
            "total_revenue": str(model.total_revenue),
            "parameter_count": model.parameter_count,
            "training_steps": model.training_steps
        }


# Global registry instance
_registry: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    """Get the global model registry instance."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
