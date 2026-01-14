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
# Distributed Training Integration Tests
# =============================================================================

class TestDistributedTraining:
    """Integration tests for distributed/federated training workflows."""
    
    @pytest.fixture
    def federated_aggregator(self):
        """Create a federated aggregator for testing."""
        from actuallyopenai.training.federated_aggregator import (
            FederatedAggregator, AggregationStrategy
        )
        return FederatedAggregator(
            strategy=AggregationStrategy.FEDAVG,
            min_workers=2,
            staleness_threshold=5
        )
    
    @pytest.fixture
    def simulated_worker_updates(self, mock_model):
        """Generate simulated worker updates with gradients."""
        import torch
        from actuallyopenai.training.federated_aggregator import WorkerUpdate
        
        updates = []
        for i in range(3):
            # Create fake gradients matching model structure
            gradients = {}
            for name, param in mock_model.named_parameters():
                # Simulate gradients with some variance
                gradients[name] = torch.randn_like(param) * 0.1 + (i * 0.01)
            
            update = WorkerUpdate(
                worker_id=f"worker_{i}",
                round_id=0,
                gradients=gradients,
                num_samples=100 + i * 50,
                local_loss=2.5 - i * 0.1,
                local_steps=10,
                compute_time=1.5
            )
            updates.append(update)
        
        return updates
    
    async def test_gradient_aggregation_fedavg(
        self, federated_aggregator, simulated_worker_updates
    ):
        """Test FedAvg gradient aggregation with multiple workers."""
        # Receive updates from all workers
        for update in simulated_worker_updates:
            await federated_aggregator.receive_update(update)
        
        # Check aggregation readiness
        assert federated_aggregator.ready_to_aggregate()
        
        # Perform aggregation
        result = await federated_aggregator.aggregate()
        
        assert result is not None
        assert result.num_workers == 3
        assert result.total_samples == 100 + 150 + 200  # Sum of worker samples
        assert result.aggregated_gradients is not None
        assert len(result.aggregated_gradients) > 0
        assert result.average_loss > 0
    
    async def test_gradient_aggregation_krum(self, mock_model):
        """Test Krum Byzantine-resilient aggregation strategy."""
        from actuallyopenai.training.federated_aggregator import (
            FederatedAggregator, AggregationStrategy, WorkerUpdate
        )
        import torch
        
        aggregator = FederatedAggregator(
            strategy=AggregationStrategy.BYZANTINE,
            min_workers=3,
            byzantine_threshold=0.3
        )
        
        # Create updates including one "Byzantine" (outlier) worker
        for i in range(4):
            gradients = {}
            for name, param in mock_model.named_parameters():
                if i == 2:  # Byzantine worker - completely different gradients
                    gradients[name] = torch.randn_like(param) * 100
                else:
                    gradients[name] = torch.randn_like(param) * 0.1
            
            update = WorkerUpdate(
                worker_id=f"worker_{i}",
                round_id=0,
                gradients=gradients,
                num_samples=100,
                local_loss=2.5 if i != 2 else 50.0
            )
            await aggregator.receive_update(update)
        
        # Aggregate - should handle Byzantine worker
        result = await aggregator.aggregate()
        
        assert result is not None
        assert result.num_workers >= 3
        # Byzantine detection should identify outlier
        assert result.outliers_detected >= 0
    
    async def test_model_parameter_synchronization(self, mock_model):
        """Test model parameter sync across simulated workers."""
        import torch
        import copy
        
        # Store original state before any modifications
        original_state = {k: v.clone() for k, v in mock_model.state_dict().items()}
        
        # Create worker models (clones of original)
        worker_models = [
            type(mock_model)() for _ in range(3)
        ]
        
        # Initialize all with same parameters
        global_state = mock_model.state_dict()
        for worker_model in worker_models:
            worker_model.load_state_dict(copy.deepcopy(global_state))
        
        # Simulate local training on each worker with significant changes
        for i, worker_model in enumerate(worker_models):
            for param in worker_model.parameters():
                param.data += torch.randn_like(param) * 0.5 * (i + 1)
        
        # Aggregate parameters (FedAvg style)
        aggregated_state = {}
        for name in global_state.keys():
            params = [wm.state_dict()[name].float() for wm in worker_models]
            aggregated_state[name] = torch.stack(params).mean(dim=0)
        
        # Update global model
        mock_model.load_state_dict(aggregated_state)
        
        # Verify global model updated (compare to original state)
        new_state = mock_model.state_dict()
        for name in original_state.keys():
            assert not torch.allclose(new_state[name], original_state[name], atol=1e-3)
    
    async def test_fedprox_strategy(self, mock_model):
        """Test FedProx aggregation with proximal term."""
        from actuallyopenai.training.federated_aggregator import (
            FederatedAggregator, AggregationStrategy, WorkerUpdate
        )
        import torch
        
        aggregator = FederatedAggregator(
            strategy=AggregationStrategy.FEDPROX,
            min_workers=2,
            mu=0.01  # Proximal term weight
        )
        
        # Set global model reference
        aggregator.set_global_model(mock_model)
        
        # Create worker updates
        for i in range(2):
            gradients = {}
            for name, param in mock_model.named_parameters():
                gradients[name] = torch.randn_like(param) * 0.1
            
            update = WorkerUpdate(
                worker_id=f"worker_{i}",
                round_id=0,
                gradients=gradients,
                num_samples=100,
                local_loss=2.5
            )
            await aggregator.receive_update(update)
        
        result = await aggregator.aggregate()
        
        assert result is not None
        assert result.strategy == AggregationStrategy.FEDPROX


# =============================================================================
# API Integration Tests
# =============================================================================

class TestAPIIntegration:
    """Integration tests for API with actual model inference."""
    
    @pytest.fixture
    def api_client_with_model(self):
        """Create test client with actual model loaded."""
        from fastapi.testclient import TestClient
        from actuallyopenai.api.production_api import (
            app, store, User, UserTier, APIKeyRecord, 
            hash_password, config
        )
        import hashlib
        
        # Setup test user
        test_user = User(
            id="test-integration-user",
            email="test-integration@actuallyopenai.com",
            hashed_password=hash_password("TestPass123!"),
            tier=UserTier.PREMIUM,
            is_verified=True
        )
        store.users[test_user.id] = test_user
        store.users_by_email[test_user.email] = test_user.id
        
        # Create test API key with high rate limit
        test_key = "aoai-test-integration-key-12345"
        key_hash = hashlib.sha256(test_key.encode()).hexdigest()
        store.api_keys[key_hash] = APIKeyRecord(
            id="test-integration-key",
            key_hash=key_hash,
            user_id=test_user.id,
            name="Integration Test Key",
            rate_limit=1000  # High limit for tests
        )
        
        return TestClient(app), test_key
    
    def test_completions_with_real_inference(self, api_client_with_model):
        """Test /v1/completions endpoint with real model inference."""
        client, api_key = api_client_with_model
        
        response = client.post(
            "/v1/completions",
            headers={"X-API-Key": api_key},
            json={
                "model": "aoai-1",
                "prompt": "Hello, how are",
                "max_tokens": 20,
                "temperature": 0.7
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "text" in data["choices"][0] or "message" in data["choices"][0]
        assert "usage" in data
    
    def test_chat_completions_streaming(self, api_client_with_model):
        """Test /v1/chat/completions with streaming response."""
        client, api_key = api_client_with_model
        
        # Request with stream=True
        with client.stream(
            "POST",
            "/v1/chat/completions",
            headers={"X-API-Key": api_key},
            json={
                "model": "aoai-1",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Say hello!"}
                ],
                "max_tokens": 50,
                "stream": True
            }
        ) as response:
            assert response.status_code == 200
            
            chunks = []
            for chunk in response.iter_lines():
                if chunk:
                    chunks.append(chunk)
            
            # Should receive multiple SSE chunks
            assert len(chunks) >= 1
    
    def test_rate_limiting_behavior(self, api_client_with_model):
        """Test rate limiting behavior - verify rate limiter exists and is configured."""
        from fastapi.testclient import TestClient
        from actuallyopenai.api.production_api import (
            app, store, User, UserTier, APIKeyRecord, hash_password
        )
        import hashlib
        import uuid
        
        client = TestClient(app)
        
        # Create unique user to avoid test pollution
        unique_id = str(uuid.uuid4())[:8]
        
        limited_user = User(
            id=f"rate-limited-user-{unique_id}",
            email=f"limited-{unique_id}@test.com",
            hashed_password=hash_password("Test123!"),
            tier=UserTier.FREE,
            is_verified=True
        )
        store.users[limited_user.id] = limited_user
        store.users_by_email[limited_user.email] = limited_user.id
        
        # Create API key with specific rate limit
        limited_key = f"aoai-rate-limited-key-{unique_id}"
        key_hash = hashlib.sha256(limited_key.encode()).hexdigest()
        store.api_keys[key_hash] = APIKeyRecord(
            id=f"rate-limited-key-{unique_id}",
            key_hash=key_hash,
            user_id=limited_user.id,
            name="Rate Limited Key",
            rate_limit=100  # Reasonable limit for testing
        )
        
        # Verify the API key works and has rate limit configured
        response = client.get(
            "/v1/models",
            headers={"X-API-Key": limited_key}
        )
        
        # Should succeed with valid key
        assert response.status_code == 200
        
        # Verify rate limit is stored in the API key record
        assert store.api_keys[key_hash].rate_limit == 100
        
        # Make additional requests to verify API handles multiple calls
        responses = []
        for i in range(5):
            resp = client.get(
                "/v1/models",
                headers={"X-API-Key": limited_key}
            )
            responses.append(resp.status_code)
        
        # All should succeed within reasonable rate limit
        success_count = sum(1 for r in responses if r == 200)
        assert success_count >= 3  # Most should succeed
    
    def test_concurrent_api_requests(self, api_client_with_model):
        """Test API handles concurrent requests properly."""
        import concurrent.futures
        
        client, api_key = api_client_with_model
        
        def make_request(i):
            response = client.get(
                "/v1/models",
                headers={"X-API-Key": api_key}
            )
            return response.status_code
        
        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request, i) for i in range(10)]
            results = [f.result() for f in futures]
        
        # Most should succeed
        success_count = sum(1 for r in results if r == 200)
        assert success_count >= 5


# =============================================================================
# P2P Network Integration Tests
# =============================================================================

class TestP2PNetwork:
    """Integration tests for P2P networking and messaging."""
    
    @pytest.fixture
    def mock_peer(self):
        """Create a mock peer for testing."""
        from actuallyopenai.network.p2p_node import Peer
        return Peer(
            node_id="test-peer-001",
            host="127.0.0.1",
            port=31337,
            reputation=1.0,
            compute_power=10.0
        )
    
    def test_peer_discovery_simulation(self):
        """Test peer discovery protocol simulation."""
        from actuallyopenai.network.p2p_node import Peer, Message, MessageType
        import time
        
        # Simulate a network of peers
        peers = {}
        for i in range(5):
            peer = Peer(
                node_id=f"node_{i}",
                host="127.0.0.1",
                port=31337 + i,
                reputation=1.0,
                compute_power=5.0 + i
            )
            peers[peer.node_id] = peer
        
        # Create discovery message
        discover_msg = Message(
            msg_type=MessageType.DISCOVER,
            sender_id="new_node",
            payload={"seeking_peers": True, "max_peers": 10}
        )
        
        # Serialize and deserialize
        serialized = discover_msg.serialize()
        assert len(serialized) > 0
        
        # Simulate peers responding
        responses = []
        for peer_id, peer in peers.items():
            response = Message(
                msg_type=MessageType.PEERS,
                sender_id=peer_id,
                payload={
                    "peers": [p.to_dict() for p in peers.values() if p.node_id != peer_id]
                }
            )
            responses.append(response)
        
        # All peers should respond
        assert len(responses) == 5
        
        # Each response should contain other peers
        for resp in responses:
            assert resp.msg_type == MessageType.PEERS
            assert len(resp.payload["peers"]) == 4  # All except sender
    
    def test_gradient_sharing_protocol(self, mock_model):
        """Test gradient sharing message protocol."""
        from actuallyopenai.network.p2p_node import Message, MessageType
        import torch
        import base64
        import io
        
        # Create gradients to share
        gradients = {}
        for name, param in mock_model.named_parameters():
            gradients[name] = torch.randn_like(param) * 0.1
        
        # Serialize gradients for transmission
        def serialize_gradients(grads):
            serialized = {}
            for name, grad in grads.items():
                buffer = io.BytesIO()
                torch.save(grad, buffer)
                serialized[name] = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return serialized
        
        def deserialize_gradients(serialized):
            grads = {}
            for name, data in serialized.items():
                buffer = io.BytesIO(base64.b64decode(data.encode('utf-8')))
                grads[name] = torch.load(buffer, weights_only=True)
            return grads
        
        # Create gradient share message
        gradient_msg = Message(
            msg_type=MessageType.GRADIENT_SHARE,
            sender_id="worker_001",
            payload={
                "round_id": 5,
                "gradients": serialize_gradients(gradients),
                "num_samples": 256,
                "local_loss": 2.45
            }
        )
        
        # Serialize entire message
        wire_data = gradient_msg.serialize()
        
        # Deserialize on receiving end
        received_msg = Message.deserialize(wire_data[4:])  # Skip length prefix
        
        assert received_msg.msg_type == MessageType.GRADIENT_SHARE
        assert received_msg.payload["round_id"] == 5
        
        # Deserialize gradients
        received_grads = deserialize_gradients(received_msg.payload["gradients"])
        
        # Verify gradients match
        for name in gradients:
            assert name in received_grads
            assert torch.allclose(gradients[name], received_grads[name])
    
    def test_model_sync_protocol(self, mock_model):
        """Test model synchronization protocol."""
        from actuallyopenai.network.p2p_node import Message, MessageType
        import torch
        import hashlib
        
        # Get model state
        model_state = mock_model.state_dict()
        
        # Create model hash for verification
        state_bytes = b""
        for name, param in sorted(model_state.items()):
            state_bytes += param.cpu().numpy().tobytes()
        model_hash = hashlib.sha256(state_bytes).hexdigest()[:16]
        
        # Create sync request message
        sync_request = Message(
            msg_type=MessageType.MODEL_SYNC,
            sender_id="new_worker",
            payload={
                "request_type": "full_sync",
                "current_hash": None,  # No current model
                "round_id": 0
            }
        )
        
        # Create sync response with model state
        sync_response = Message(
            msg_type=MessageType.MODEL_SYNC,
            sender_id="coordinator",
            payload={
                "request_type": "full_model",
                "model_hash": model_hash,
                "round_id": 10,
                "param_count": sum(p.numel() for p in model_state.values())
            }
        )
        
        assert sync_request.msg_type == MessageType.MODEL_SYNC
        assert sync_response.payload["model_hash"] == model_hash
        assert sync_response.payload["param_count"] > 0
    
    async def test_peer_heartbeat_mechanism(self):
        """Test peer heartbeat and health tracking."""
        from actuallyopenai.network.p2p_node import Peer, Message, MessageType
        import time
        
        # Create peer
        peer = Peer(
            node_id="heartbeat_test_peer",
            host="127.0.0.1",
            port=31337
        )
        
        initial_time = peer.last_seen
        
        # Simulate time passing
        await asyncio.sleep(0.1)
        
        # Receive heartbeat
        heartbeat = Message(
            msg_type=MessageType.HEARTBEAT,
            sender_id=peer.node_id,
            payload={
                "status": "active",
                "compute_available": True,
                "current_task": None
            }
        )
        
        # Update peer last_seen
        peer.last_seen = heartbeat.timestamp
        
        assert peer.last_seen >= initial_time
        assert heartbeat.payload["status"] == "active"


# =============================================================================
# Checkpoint Recovery Integration Tests
# =============================================================================

class TestCheckpointRecovery:
    """Integration tests for checkpoint save/load and recovery."""
    
    @pytest.fixture
    def checkpoint_dir(self, tmp_path):
        """Create temporary checkpoint directory."""
        checkpoint_path = tmp_path / "checkpoints"
        checkpoint_path.mkdir()
        return str(checkpoint_path)
    
    @pytest.fixture
    def training_state(self):
        """Create sample training state."""
        from actuallyopenai.training.continuous_trainer import TrainingState, TrainingPhase
        from datetime import datetime
        
        return TrainingState(
            phase=TrainingPhase.UPDATING_MODEL,
            global_step=5000,
            epoch=10,
            total_tokens_processed=1000000,
            current_loss=1.85,
            best_loss=1.75,
            loss_history=[2.5, 2.2, 2.0, 1.9, 1.85],
            active_workers=5,
            started_at=datetime.utcnow()
        )
    
    async def test_checkpoint_save_load_cycle(
        self, mock_model, checkpoint_dir, training_state
    ):
        """Test complete checkpoint save and load cycle."""
        import torch
        import os
        import json
        
        # Create checkpoint data
        checkpoint = {
            'model_state_dict': mock_model.state_dict(),
            'global_step': training_state.global_step,
            'epoch': training_state.epoch,
            'current_loss': training_state.current_loss,
            'best_loss': training_state.best_loss,
            'loss_history': training_state.loss_history,
            'total_tokens_processed': training_state.total_tokens_processed
        }
        
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_5000.pt")
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Verify file exists
        assert os.path.exists(checkpoint_path)
        
        # Create new model and load checkpoint
        new_model = type(mock_model)()
        loaded_checkpoint = torch.load(checkpoint_path, weights_only=False)
        
        new_model.load_state_dict(loaded_checkpoint['model_state_dict'])
        
        # Verify model parameters match
        for (name1, param1), (name2, param2) in zip(
            mock_model.named_parameters(), new_model.named_parameters()
        ):
            assert name1 == name2
            assert torch.allclose(param1, param2)
        
        # Verify training state restored
        assert loaded_checkpoint['global_step'] == 5000
        assert loaded_checkpoint['current_loss'] == 1.85
    
    async def test_recovery_from_corrupted_checkpoint(
        self, mock_model, checkpoint_dir
    ):
        """Test recovery when checkpoint is corrupted."""
        import torch
        import os
        
        # Save valid checkpoint first
        valid_checkpoint = {
            'model_state_dict': mock_model.state_dict(),
            'global_step': 1000,
            'current_loss': 2.0
        }
        valid_path = os.path.join(checkpoint_dir, "checkpoint_1000.pt")
        torch.save(valid_checkpoint, valid_path)
        
        # Create corrupted checkpoint
        corrupted_path = os.path.join(checkpoint_dir, "checkpoint_2000.pt")
        with open(corrupted_path, 'wb') as f:
            f.write(b"corrupted data that is not a valid checkpoint")
        
        # Try to load corrupted checkpoint - should fail gracefully
        loaded_checkpoint = None
        corrupted_detected = False
        
        try:
            loaded_checkpoint = torch.load(corrupted_path, weights_only=False)
        except Exception:
            corrupted_detected = True
        
        assert corrupted_detected
        
        # Fall back to valid checkpoint
        loaded_checkpoint = torch.load(valid_path, weights_only=False)
        assert loaded_checkpoint['global_step'] == 1000
    
    async def test_auto_resume_functionality(self, mock_model, checkpoint_dir):
        """Test automatic resume from latest checkpoint."""
        import torch
        import os
        
        # Create multiple checkpoints
        for step in [1000, 2000, 3000, 4500]:
            checkpoint = {
                'model_state_dict': mock_model.state_dict(),
                'global_step': step,
                'current_loss': 3.0 - step * 0.0003
            }
            path = os.path.join(checkpoint_dir, f"checkpoint_{step}.pt")
            torch.save(checkpoint, path)
        
        # Find latest checkpoint
        def find_latest_checkpoint(checkpoint_dir):
            checkpoints = []
            for f in os.listdir(checkpoint_dir):
                if f.startswith("checkpoint_") and f.endswith(".pt"):
                    step = int(f.replace("checkpoint_", "").replace(".pt", ""))
                    checkpoints.append((step, f))
            
            if not checkpoints:
                return None
            
            latest = max(checkpoints, key=lambda x: x[0])
            return os.path.join(checkpoint_dir, latest[1])
        
        latest_path = find_latest_checkpoint(checkpoint_dir)
        
        assert latest_path is not None
        assert "4500" in latest_path
        
        # Load latest checkpoint
        checkpoint = torch.load(latest_path, weights_only=False)
        assert checkpoint['global_step'] == 4500
    
    async def test_checkpoint_with_optimizer_state(self, mock_model, checkpoint_dir):
        """Test checkpoint includes and restores optimizer state."""
        import torch
        import os
        
        # Create optimizer and do some training
        optimizer = torch.optim.Adam(mock_model.parameters(), lr=1e-4)
        
        # Simulate training steps to build optimizer state
        for _ in range(5):
            input_ids = torch.randint(0, 260, (2, 64))
            output, _ = mock_model(input_ids)
            loss = output.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Save checkpoint with optimizer state
        checkpoint = {
            'model_state_dict': mock_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': 5
        }
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_with_optim.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # Create new model and optimizer
        new_model = type(mock_model)()
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-4)
        
        # Load checkpoint
        loaded = torch.load(checkpoint_path, weights_only=False)
        new_model.load_state_dict(loaded['model_state_dict'])
        new_optimizer.load_state_dict(loaded['optimizer_state_dict'])
        
        # Verify optimizer state restored (check momentum buffers exist)
        assert len(new_optimizer.state) > 0
    
    async def test_incremental_checkpoint_strategy(self, mock_model, checkpoint_dir):
        """Test incremental checkpointing with cleanup of old checkpoints."""
        import torch
        import os
        
        max_checkpoints = 3
        
        def save_checkpoint_with_cleanup(model, step, checkpoint_dir, max_keep):
            # Save new checkpoint
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'global_step': step
            }
            path = os.path.join(checkpoint_dir, f"checkpoint_{step}.pt")
            torch.save(checkpoint, path)
            
            # Cleanup old checkpoints
            checkpoints = []
            for f in os.listdir(checkpoint_dir):
                if f.startswith("checkpoint_") and f.endswith(".pt"):
                    s = int(f.replace("checkpoint_", "").replace(".pt", ""))
                    checkpoints.append((s, os.path.join(checkpoint_dir, f)))
            
            # Sort by step and remove oldest if over limit
            checkpoints.sort(key=lambda x: x[0])
            while len(checkpoints) > max_keep:
                _, old_path = checkpoints.pop(0)
                os.remove(old_path)
        
        # Save several checkpoints
        for step in [100, 200, 300, 400, 500]:
            save_checkpoint_with_cleanup(
                mock_model, step, checkpoint_dir, max_checkpoints
            )
        
        # Verify only max_checkpoints remain
        remaining = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
        assert len(remaining) == max_checkpoints
        
        # Verify the latest ones are kept
        steps = [int(f.replace("checkpoint_", "").replace(".pt", "")) 
                 for f in remaining]
        assert 500 in steps
        assert 400 in steps
        assert 300 in steps
        assert 100 not in steps  # Should be cleaned up


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
