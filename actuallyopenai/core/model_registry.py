"""
Model Registry - Decentralized storage and management of trained AI models.
Supports database persistence with in-memory cache fallback.
"""

import hashlib
import json
import os
from datetime import datetime
from decimal import Decimal
from typing import Optional, List, Dict, Any
import uuid

import structlog
from pydantic import BaseModel
from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker

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


class ModelVersionDB:
    """Simple key-value store for model versions using JSON files."""
    
    def __init__(self, storage_path: str = "aoai_data/model_versions"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
    
    def _get_file_path(self, model_id: str) -> str:
        return os.path.join(self.storage_path, f"{model_id}_versions.json")
    
    def save_versions(self, model_id: str, versions: List[ModelVersion]):
        """Save model versions to file."""
        try:
            file_path = self._get_file_path(model_id)
            data = [v.model_dump() for v in versions]
            # Convert datetime to ISO format for JSON serialization
            for item in data:
                if isinstance(item.get('created_at'), datetime):
                    item['created_at'] = item['created_at'].isoformat()
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.warning("Failed to save model versions", model_id=model_id, error=str(e))
    
    def load_versions(self, model_id: str) -> List[ModelVersion]:
        """Load model versions from file."""
        try:
            file_path = self._get_file_path(model_id)
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                return [ModelVersion(**item) for item in data]
        except Exception as e:
            logger.warning("Failed to load model versions", model_id=model_id, error=str(e))
        return []


class ModelRegistry:
    """
    Registry for managing trained AI models.
    Supports IPFS storage for decentralized model hosting.
    Uses database persistence with in-memory cache fallback.
    """
    
    def __init__(self, ipfs_host: str = None, ipfs_port: int = None, session_factory=None):
        settings = get_settings()
        self.ipfs_host = ipfs_host or settings.ipfs_host
        self.ipfs_port = ipfs_port or settings.ipfs_port
        
        # Database session factory (optional)
        self._session_factory = session_factory
        self._db_available = False
        
        # In-memory cache (used when DB unavailable or for speed)
        self.models: Dict[str, ModelInfo] = {}
        self.model_versions: Dict[str, List[ModelVersion]] = {}
        
        # File-based version storage (fallback for versions)
        self._version_storage = ModelVersionDB()
        
        # IPFS client (lazy initialization)
        self._ipfs_client = None
        
        # Load cached data on init
        self._load_from_cache()
    
    def _load_from_cache(self):
        """Load models from local JSON cache on startup."""
        cache_file = "aoai_data/model_registry_cache.json"
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                for model_data in data.get('models', []):
                    try:
                        model = ModelInfo(**model_data)
                        self.models[model.id] = model
                        # Load versions for this model
                        self.model_versions[model.id] = self._version_storage.load_versions(model.id)
                    except Exception as e:
                        logger.warning("Failed to load cached model", error=str(e))
                logger.info("Loaded models from cache", count=len(self.models))
        except Exception as e:
            logger.warning("Failed to load model registry cache", error=str(e))
    
    def _save_to_cache(self):
        """Save models to local JSON cache."""
        cache_file = "aoai_data/model_registry_cache.json"
        try:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            data = {
                'models': [m.model_dump() for m in self.models.values()],
                'updated_at': datetime.utcnow().isoformat()
            }
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.warning("Failed to save model registry cache", error=str(e))
    
    async def _get_db_session(self) -> Optional[AsyncSession]:
        """Get database session if available."""
        if self._session_factory:
            try:
                return self._session_factory()
            except Exception as e:
                logger.warning("Failed to create DB session", error=str(e))
        return None
    
    async def _sync_to_database(self, model: ModelInfo):
        """Sync model to database if available."""
        session = await self._get_db_session()
        if not session:
            return
        
        try:
            from actuallyopenai.core.database import ModelDB
            
            async with session.begin():
                # Check if model exists
                result = await session.execute(
                    select(ModelDB).where(ModelDB.id == model.id)
                )
                existing = result.scalar_one_or_none()
                
                if existing:
                    # Update existing model
                    await session.execute(
                        update(ModelDB).where(ModelDB.id == model.id).values(
                            name=model.name,
                            description=model.description,
                            model_type=model.model_type,
                            architecture=model.architecture,
                            parameter_count=model.parameter_count,
                            file_size_bytes=model.file_size_bytes,
                            ipfs_hash=model.ipfs_hash,
                            training_steps=model.training_steps,
                            final_loss=model.final_loss,
                            version=model.version,
                            is_public=model.is_public,
                            api_calls=model.api_calls,
                            total_revenue=model.total_revenue,
                            updated_at=datetime.utcnow()
                        )
                    )
                else:
                    # Insert new model
                    db_model = ModelDB(
                        id=model.id,
                        name=model.name,
                        description=model.description,
                        model_type=model.model_type,
                        architecture=model.architecture,
                        parameter_count=model.parameter_count,
                        file_size_bytes=model.file_size_bytes,
                        ipfs_hash=model.ipfs_hash,
                        training_steps=model.training_steps,
                        final_loss=model.final_loss,
                        version=model.version,
                        is_public=model.is_public,
                        api_calls=model.api_calls,
                        total_revenue=model.total_revenue
                    )
                    session.add(db_model)
            
            self._db_available = True
            logger.debug("Model synced to database", model_id=model.id)
        except Exception as e:
            self._db_available = False
            logger.warning("Failed to sync model to database", model_id=model.id, error=str(e))
        finally:
            await session.close()
    
    async def _load_from_database(self):
        """Load all models from database."""
        session = await self._get_db_session()
        if not session:
            return
        
        try:
            from actuallyopenai.core.database import ModelDB
            
            async with session.begin():
                result = await session.execute(select(ModelDB))
                db_models = result.scalars().all()
                
                for db_model in db_models:
                    model = ModelInfo(
                        id=db_model.id,
                        name=db_model.name,
                        description=db_model.description or "",
                        model_type=db_model.model_type,
                        architecture=db_model.architecture,
                        parameter_count=db_model.parameter_count or 0,
                        file_size_bytes=db_model.file_size_bytes or 0,
                        ipfs_hash=db_model.ipfs_hash,
                        training_steps=db_model.training_steps or 0,
                        final_loss=db_model.final_loss,
                        version=db_model.version or "1.0.0",
                        is_public=db_model.is_public if db_model.is_public is not None else True,
                        api_calls=db_model.api_calls or 0,
                        total_revenue=db_model.total_revenue or Decimal("0"),
                        created_at=db_model.created_at or datetime.utcnow(),
                        updated_at=db_model.updated_at or datetime.utcnow()
                    )
                    self.models[model.id] = model
                    # Load versions
                    self.model_versions[model.id] = self._version_storage.load_versions(model.id)
            
            self._db_available = True
            logger.info("Loaded models from database", count=len(db_models))
            # Update local cache
            self._save_to_cache()
        except Exception as e:
            self._db_available = False
            logger.warning("Failed to load models from database", error=str(e))
        finally:
            await session.close()
    
    def set_session_factory(self, session_factory):
        """Set the database session factory."""
        self._session_factory = session_factory
    
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
        
        # Persist to database
        await self._sync_to_database(model)
        # Save to local cache
        self._save_to_cache()
        
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
        
        # Persist changes
        await self._sync_to_database(model)
        self._version_storage.save_versions(model_id, self.model_versions[model_id])
        self._save_to_cache()
        
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
            # Periodically sync to database (every 100 calls)
            if self.models[model_id].api_calls % 100 == 0:
                await self._sync_to_database(self.models[model_id])
                self._save_to_cache()
    
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
    
    async def delete_model(self, model_id: str) -> bool:
        """Delete a model from the registry."""
        if model_id not in self.models:
            return False
        
        # Remove from memory
        del self.models[model_id]
        if model_id in self.model_versions:
            del self.model_versions[model_id]
        
        # Remove from database if available
        session = await self._get_db_session()
        if session:
            try:
                from actuallyopenai.core.database import ModelDB
                
                async with session.begin():
                    await session.execute(
                        delete(ModelDB).where(ModelDB.id == model_id)
                    )
                logger.info("Model deleted from database", model_id=model_id)
            except Exception as e:
                logger.warning("Failed to delete model from database", model_id=model_id, error=str(e))
            finally:
                await session.close()
        
        # Update cache
        self._save_to_cache()
        
        return True
    
    async def initialize(self):
        """Initialize the registry - load from database if available."""
        await self._load_from_database()


# Global registry instance
_registry: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    """Get the global model registry instance."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


async def initialize_model_registry(session_factory=None) -> ModelRegistry:
    """Initialize the model registry with optional database support."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry(session_factory=session_factory)
    elif session_factory:
        _registry.set_session_factory(session_factory)
    await _registry.initialize()
    return _registry
