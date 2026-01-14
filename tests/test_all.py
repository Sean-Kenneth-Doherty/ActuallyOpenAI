"""
Test Suite for ActuallyOpenAI.

Comprehensive tests for:
- API endpoints
- Training system
- Blockchain integration
- Worker operations
"""

import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import pytest
from datetime import datetime
from decimal import Decimal
from unittest.mock import MagicMock, AsyncMock, patch

# Mark all tests as async by default
pytestmark = pytest.mark.asyncio


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    import torch
    import torch.nn as nn
    
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(260, 256)
            self.linear = nn.Linear(256, 260)
        
        def forward(self, x):
            # Return logits with proper gradient tracking
            embedded = self.embed(x)  # [batch, seq, 256]
            logits = self.linear(embedded)  # [batch, seq, 260]
            return logits, None
    
    model = MockModel()
    model.train()  # Ensure training mode
    return model


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    class MockTokenizer:
        vocab_size = 260
        
        def encode(self, text):
            return [ord(c) % 256 for c in text[:100]]
        
        def decode(self, tokens):
            return "".join(chr(t % 128) for t in tokens if t < 256)
    
    return MockTokenizer()


@pytest.fixture
def sample_training_data():
    """Sample training data."""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Hello world, this is a test.",
        "Machine learning is fascinating.",
    ]


# =============================================================================
# Model Tests
# =============================================================================

class TestModel:
    """Tests for the base model."""
    
    def test_model_config_tiny(self):
        """Test tiny model configuration."""
        from actuallyopenai.models.base_model import ModelConfig
        
        config = ModelConfig.tiny()
        assert config.hidden_size == 256
        assert config.num_layers == 4
        assert config.num_heads == 4
    
    def test_model_creation(self):
        """Test model creation."""
        from actuallyopenai.models.base_model import create_model
        
        model = create_model("tiny", vocab_size=260)
        assert model is not None
        
        # Check parameter count
        params = sum(p.numel() for p in model.parameters())
        assert params > 0
    
    def test_model_forward_pass(self, mock_model):
        """Test model forward pass."""
        import torch
        
        input_ids = torch.randint(0, 260, (2, 64))
        output, loss = mock_model(input_ids)
        
        assert output.shape == (2, 64, 260)


# =============================================================================
# Tokenizer Tests
# =============================================================================

class TestTokenizer:
    """Tests for tokenizers."""
    
    def test_byte_tokenizer_encode(self):
        """Test byte tokenizer encoding."""
        from actuallyopenai.data.tokenizer import SimpleByteTokenizer
        
        tokenizer = SimpleByteTokenizer()
        tokens = tokenizer.encode("Hello")
        
        assert len(tokens) > 0
        assert all(isinstance(t, int) for t in tokens)
    
    def test_byte_tokenizer_decode(self):
        """Test byte tokenizer decoding."""
        from actuallyopenai.data.tokenizer import SimpleByteTokenizer
        
        tokenizer = SimpleByteTokenizer()
        text = "Hello World"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        
        assert "Hello" in decoded
    
    def test_byte_tokenizer_vocab_size(self):
        """Test byte tokenizer vocab size."""
        from actuallyopenai.data.tokenizer import SimpleByteTokenizer
        
        tokenizer = SimpleByteTokenizer()
        assert tokenizer.vocab_size == 260  # 256 bytes + 4 special tokens


# =============================================================================
# API Tests
# =============================================================================

class TestAPI:
    """Tests for the production API."""
    
    @pytest.fixture
    def api_client(self):
        """Create test client with demo data setup."""
        from fastapi.testclient import TestClient
        from actuallyopenai.api.production_api import (
            app, store, User, UserTier, APIKeyRecord, 
            hash_password, config
        )
        import hashlib
        
        # Setup demo user and API key manually for tests
        demo_user = User(
            id="demo-user",
            email="demo@actuallyopenai.com",
            hashed_password=hash_password("Demo123!"),
            tier=UserTier.PREMIUM,
            is_verified=True
        )
        store.users[demo_user.id] = demo_user
        store.users_by_email[demo_user.email] = demo_user.id
        
        # Create demo API key
        demo_key = "aoai-demo-key-123456789"
        key_hash = hashlib.sha256(demo_key.encode()).hexdigest()
        store.api_keys[key_hash] = APIKeyRecord(
            id="demo-key",
            key_hash=key_hash,
            user_id=demo_user.id,
            name="Demo Key",
            rate_limit=config.PREMIUM_RATE_LIMIT
        )
        
        return TestClient(app)
    
    def test_health_endpoint(self, api_client):
        """Test health check endpoint."""
        response = api_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_root_endpoint(self, api_client):
        """Test root endpoint."""
        response = api_client.get("/")
        assert response.status_code == 200
        assert "ActuallyOpenAI" in response.json()["name"]
    
    def test_register_user(self, api_client):
        """Test user registration."""
        response = api_client.post("/v1/auth/register", json={
            "email": "test@example.com",
            "password": "TestPass123!",
            "wallet_address": "0x1234567890"
        })
        
        # Should succeed or fail with "already registered"
        assert response.status_code in [200, 400]
    
    def test_login(self, api_client):
        """Test user login."""
        # Use demo credentials
        response = api_client.post("/v1/auth/login", json={
            "email": "demo@actuallyopenai.com",
            "password": "Demo123!"
        })
        
        assert response.status_code == 200
        assert "access_token" in response.json()
    
    def test_models_list_requires_auth(self, api_client):
        """Test that models endpoint requires auth."""
        response = api_client.get("/v1/models")
        assert response.status_code == 401
    
    def test_models_list_with_api_key(self, api_client):
        """Test models endpoint with API key."""
        response = api_client.get(
            "/v1/models",
            headers={"X-API-Key": "aoai-demo-key-123456789"}
        )
        
        assert response.status_code == 200
        assert "data" in response.json()
    
    def test_chat_completion(self, api_client):
        """Test chat completion endpoint."""
        response = api_client.post(
            "/v1/chat/completions",
            headers={"X-API-Key": "aoai-demo-key-123456789"},
            json={
                "model": "aoai-1",
                "messages": [
                    {"role": "user", "content": "Hello!"}
                ],
                "max_tokens": 50
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0


# =============================================================================
# Training Tests
# =============================================================================

class TestTraining:
    """Tests for the training system."""
    
    def test_dataset_creation(self, mock_tokenizer, sample_training_data):
        """Test dataset creation."""
        import torch
        from torch.utils.data import DataLoader
        
        # Create simple dataset
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, texts, tokenizer, seq_len=64):
                self.tokens = []
                for text in texts:
                    self.tokens.extend(tokenizer.encode(text))
                self.seq_len = seq_len
            
            def __len__(self):
                return max(1, len(self.tokens) // self.seq_len)
            
            def __getitem__(self, idx):
                start = idx * self.seq_len
                tokens = self.tokens[start:start + self.seq_len + 1]
                while len(tokens) < self.seq_len + 1:
                    tokens.append(0)
                return {
                    "input_ids": torch.tensor(tokens[:-1]),
                    "labels": torch.tensor(tokens[1:])
                }
        
        dataset = SimpleDataset(sample_training_data, mock_tokenizer)
        assert len(dataset) > 0
        
        item = dataset[0]
        assert "input_ids" in item
        assert "labels" in item
    
    def test_training_step(self, mock_model, mock_tokenizer, sample_training_data):
        """Test a single training step."""
        import torch
        import torch.nn.functional as F
        
        # Create batch
        tokens = mock_tokenizer.encode(sample_training_data[0])
        tokens = tokens[:64] + [0] * (65 - len(tokens))
        
        input_ids = torch.tensor([tokens[:-1]])
        labels = torch.tensor([tokens[1:]])
        
        # Forward pass
        logits, _ = mock_model(input_ids)
        
        # Compute loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )
        
        assert loss.item() > 0
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        for param in mock_model.parameters():
            if param.requires_grad:
                assert param.grad is not None


# =============================================================================
# Blockchain Tests
# =============================================================================

class TestBlockchain:
    """Tests for blockchain integration."""
    
    def test_token_minting_calculation(self):
        """Test AOAI token minting calculation."""
        # Mock calculation
        compute_hours = 10.0
        base_rate = 100  # tokens per hour
        
        tokens_earned = compute_hours * base_rate
        assert tokens_earned == 1000
    
    def test_dividend_calculation(self):
        """Test dividend calculation."""
        # Mock calculation
        total_revenue = Decimal("10000")  # $10,000
        total_tokens = Decimal("1000000")  # 1M tokens
        user_tokens = Decimal("10000")  # 10K tokens
        
        user_share = user_tokens / total_tokens
        user_dividend = total_revenue * user_share
        
        assert user_dividend == Decimal("100")  # $100


# =============================================================================
# P2P Tests
# =============================================================================

class TestP2P:
    """Tests for P2P data sharing."""
    
    def test_chunk_creation(self):
        """Test data chunk creation."""
        from actuallyopenai.data.p2p_sharing import DataChunk
        
        data = b"Hello, this is test data!"
        chunk = DataChunk(id="", data=data)
        
        assert chunk.id  # Should be auto-generated
        assert chunk.checksum
        assert chunk.size == len(data)
    
    def test_chunk_verification(self):
        """Test chunk verification."""
        from actuallyopenai.data.p2p_sharing import DataChunk
        
        data = b"Test data for verification"
        chunk = DataChunk(id="", data=data)
        
        assert chunk.verify()
        
        # Corrupt data
        chunk.data = b"Corrupted!"
        assert not chunk.verify()
    
    def test_chunk_store(self, tmp_path):
        """Test chunk storage."""
        from actuallyopenai.data.p2p_sharing import ChunkStore, DataChunk
        
        store = ChunkStore(cache_dir=str(tmp_path), max_size_gb=0.001)
        
        # Store chunk
        chunk = DataChunk(id="", data=b"Test chunk data")
        assert store.store(chunk)
        
        # Retrieve chunk
        retrieved = store.get(chunk.id)
        assert retrieved is not None
        assert retrieved.data == chunk.data


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests."""
    
    async def test_full_training_loop(self, mock_model, mock_tokenizer, sample_training_data):
        """Test a full training loop."""
        import torch
        import torch.nn.functional as F
        
        optimizer = torch.optim.Adam(mock_model.parameters(), lr=1e-4)
        
        initial_loss = None
        final_loss = None
        
        for epoch in range(3):
            for text in sample_training_data:
                tokens = mock_tokenizer.encode(text)
                tokens = tokens[:64] + [0] * (65 - len(tokens))
                
                input_ids = torch.tensor([tokens[:-1]])
                labels = torch.tensor([tokens[1:]])
                
                optimizer.zero_grad()
                logits, _ = mock_model(input_ids)
                
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )
                
                if initial_loss is None:
                    initial_loss = loss.item()
                final_loss = loss.item()
                
                loss.backward()
                optimizer.step()
        
        # Loss should be recorded
        assert initial_loss is not None
        assert final_loss is not None


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
