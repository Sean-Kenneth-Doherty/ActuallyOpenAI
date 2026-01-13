# ActuallyOpenAI - Distributed AI Training Network

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch 2.0+">
  <img src="https://img.shields.io/badge/FastAPI-0.100+-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/Docker-Ready-blue.svg" alt="Docker Ready">
  <img src="https://img.shields.io/github/stars/actuallyopenai/actuallyopenai?style=social" alt="GitHub Stars">
</p>

<p align="center">
  <b>The AI that belongs to everyone. Train it. Own it. Earn from it.</b>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-how-it-works">How It Works</a> •
  <a href="#-api-usage">API Usage</a> •
  <a href="#-contribute-compute">Contribute Compute</a> •
  <a href="#-roadmap">Roadmap</a>
</p>

---

## 🌍 Vision

ActuallyOpenAI is a truly decentralized, crowd-funded AI training platform where anyone in the world can contribute their compute power and receive dividends from the AI products built on the network.

**Core Principles:**
- 🔓 **Open Source** - All code is transparent and auditable
- 🌐 **Distributed** - No single point of control
- 💰 **Fair Rewards** - Contributors get paid based on actual compute contribution
- 🤝 **Community Owned** - The AI belongs to those who build it

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      ActuallyOpenAI Network                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Worker 1   │    │   Worker 2   │    │   Worker N   │       │
│  │  (GPU/CPU)   │    │  (GPU/CPU)   │    │  (GPU/CPU)   │       │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘       │
│         │                   │                   │                │
│         └───────────────────┼───────────────────┘                │
│                             │                                    │
│                    ┌────────▼────────┐                          │
│                    │   Orchestrator   │                          │
│                    │     Server       │                          │
│                    └────────┬────────┘                          │
│                             │                                    │
│         ┌───────────────────┼───────────────────┐               │
│         │                   │                   │                │
│  ┌──────▼──────┐    ┌──────▼──────┐    ┌──────▼──────┐        │
│  │ Contribution │    │   Model     │    │  Dividend   │        │
│  │   Tracker    │    │  Registry   │    │   System    │        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
│                                                                  │
│                    ┌─────────────────┐                          │
│                    │   Product API   │ ◄── Revenue Stream       │
│                    │  (Monetization) │                          │
│                    └─────────────────┘                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/actuallyopenai/actuallyopenai.git
cd actuallyopenai

# Configure environment
cp .env.docker .env
# Edit .env with your wallet address and settings

# Start the full stack
docker compose up -d

# Start a worker (CPU)
docker compose --profile worker up -d

# Start a worker (GPU - requires NVIDIA Docker)
docker compose --profile worker-gpu up -d
```

### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/actuallyopenai/actuallyopenai.git
cd actuallyopenai

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -e .

# Setup environment
cp .env.example .env
# Edit .env with your configuration
```

### CLI Commands

```bash
# Show help and info
aoai info

# Start the orchestrator server
aoai serve orchestrator

# Start the model API server  
aoai serve api

# Start a worker node (earn AOAI tokens!)
aoai worker start --wallet YOUR_WALLET_ADDRESS

# View network statistics
aoai stats

# View top contributors
aoai leaderboard

# Check wallet balance
aoai wallet balance YOUR_WALLET_ADDRESS
```

## 📦 Components

### 1. Orchestrator Server
The central coordination hub that:
- Distributes training tasks to workers
- Aggregates model gradients (federated learning)
- Manages the training pipeline
- Handles worker registration and health checks

### 2. Worker Node
The compute contribution client that:
- Connects to the orchestrator
- Downloads model shards and training data
- Performs local training iterations
- Reports compute metrics for reward calculation

### 3. Contribution Tracker
Blockchain-based tracking system that:
- Records all compute contributions immutably
- Calculates contribution scores
- Provides transparent audit trail

### 4. Dividend System
Fair reward distribution that:
- Collects revenue from API usage
- Calculates contributor shares
- Distributes dividends periodically

### 5. Model Registry
Decentralized model storage that:
- Stores trained model checkpoints
- Manages model versions
- Provides model access control

### 6. Product API
Monetization layer that:
- Exposes trained models via API
- Handles authentication and billing
- Generates revenue for the collective

## 💰 How AOAI Token & Dividends Work

### The AOAI Token
AOAI is an ERC-20 token on Ethereum that:
- Is **minted** as rewards for compute contributions
- Entitles holders to **dividends** from API revenue
- Has a **max supply** of 1 billion tokens
- Is **transparent** - all minting is on-chain

### Earning AOAI
1. **Run a Worker Node**: Connect your GPU/CPU to the network
2. **Complete Training Tasks**: Process batches assigned by the orchestrator
3. **Receive AOAI**: Tokens are minted to your wallet based on:
   - Compute time provided
   - GPU hours contributed
   - Task completion quality

### Dividend Distribution
```
┌─────────────────────────────────────────────────────────┐
│                    Revenue Flow                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│    API Users ──► Model API ──► Revenue Pool             │
│                                    │                     │
│                                    ▼                     │
│              ┌─────────────────────────────┐            │
│              │   AOAI Token Contract       │            │
│              │   depositDividends()        │            │
│              └─────────────┬───────────────┘            │
│                            │                             │
│         ┌──────────────────┼──────────────────┐         │
│         ▼                  ▼                  ▼          │
│    ┌─────────┐       ┌─────────┐       ┌─────────┐     │
│    │ Holder A│       │ Holder B│       │ Holder C│     │
│    │  10% ──►│       │  30% ──►│       │  60% ──►│     │
│    │ tokens  │       │ tokens  │       │ tokens  │     │
│    └─────────┘       └─────────┘       └─────────┘     │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

Your dividend share = `(Your AOAI Balance / Total AOAI Supply) × Revenue`

### Claiming Dividends
```bash
# Check pending dividends
aoai wallet balance YOUR_WALLET

# Claim via smart contract (or UI when available)
aoai wallet claim YOUR_WALLET --key YOUR_PRIVATE_KEY
```

## 🔐 Security

- End-to-end encryption for all communications
- Secure gradient aggregation (prevents model poisoning)
- Wallet-based authentication
- Byzantine fault tolerance for distributed consensus

## 🛣️ Roadmap

- [x] Core orchestrator implementation
- [x] Worker node client
- [x] Basic contribution tracking
- [x] AOAI Token smart contract (ERC-20 with dividends)
- [x] Model API for monetization
- [x] CLI tools
- [x] Docker deployment
- [x] **Self-Improving AI Training System**
  - [x] Continuous Trainer - Never stops improving
  - [x] Federated Learning Aggregator
  - [x] Model Evolution & Generations
  - [x] Auto-Scaling Based on Compute
  - [x] Benchmark & Improvement Tracking
- [ ] Production smart contract deployment
- [ ] Web dashboard for contributors
- [ ] Multi-model training pipelines
- [ ] Mobile worker app
- [ ] DAO governance for network decisions

## 🧠 The Self-Assembling AI

What makes ActuallyOpenAI truly unique is its **self-improving architecture**. The AI doesn't just train once - it continuously evolves and improves as more compute is contributed.

### How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                  Self-Improving Training Loop                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│    Workers Join ──► More Compute ──► Faster Training            │
│         │                                    │                   │
│         │              ┌─────────────────────┘                   │
│         │              ▼                                         │
│    ┌────▼────┐   ┌──────────────┐   ┌───────────────┐          │
│    │ Gradient │   │  Federated   │   │    Model      │          │
│    │ Compute  │──►│  Aggregation │──►│   Evolution   │          │
│    └─────────┘   └──────────────┘   └───────┬───────┘          │
│                                              │                   │
│                        ┌─────────────────────┘                   │
│                        ▼                                         │
│              ┌──────────────────┐                               │
│              │   Benchmarking   │                               │
│              │  & Improvement   │                               │
│              │    Tracking      │                               │
│              └────────┬─────────┘                               │
│                       │                                          │
│         ┌─────────────┴─────────────┐                           │
│         ▼                           ▼                            │
│    Better Model             Auto-Scaling                        │
│    Released! 🎉             (adjust to compute)                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

1. **Continuous Trainer** (`actuallyopenai/training/continuous_trainer.py`)
   - Runs 24/7, never stops training
   - Dynamically adapts to available workers
   - Automatically checkpoints progress
   - Tracks all improvements over time

2. **Federated Aggregator** (`actuallyopenai/training/federated_aggregator.py`)
   - Combines gradients from all workers
   - Multiple strategies: FedAvg, FedProx, FedAdam, Byzantine-tolerant
   - Handles worker quality scoring
   - Detects and filters malicious workers

3. **Model Evolution** (`actuallyopenai/training/model_evolution.py`)
   - Tracks model generations over time
   - Maintains lineage (which model came from which)
   - Supports branching for experiments
   - Tournament selection for best models

4. **Auto-Scaler** (`actuallyopenai/training/auto_scaler.py`)
   - Monitors available compute in real-time
   - Automatically adjusts batch size, learning rate
   - Multiple modes: aggressive, balanced, conservative, adaptive
   - Handles workers joining/leaving gracefully

5. **Improvement Tracker** (`actuallyopenai/training/improvement_tracker.py`)
   - Runs standard benchmarks (perplexity, accuracy, reasoning)
   - Tracks improvement over time
   - Maintains leaderboard of best generations
   - Measures efficiency (improvement per compute hour)

### The Result: An AI That Gets Better Over Time

```
Generation 1: Loss 5.2 (100 compute hours)
     │
     └──► Generation 2: Loss 4.8 (+7.7% improvement)
              │
              └──► Generation 3: Loss 4.3 (+10.4% improvement)
                       │
                       └──► Generation 4: Loss 3.9 (+9.3% improvement)
                                │
                                └──► ... continues forever!
```

**The more people contribute compute, the better the AI becomes, and the more valuable the AOAI token becomes.**

## 📁 Project Structure

```
actuallyopenai/
├── actuallyopenai/           # Main Python package
│   ├── api/                  # Model serving API
│   ├── blockchain/           # Token & smart contract integration
│   ├── cli/                  # Command-line interface
│   ├── core/                 # Shared models & database
│   ├── orchestrator/         # Central coordination server
│   └── worker/               # Compute contribution client
├── contracts/                # Solidity smart contracts
│   ├── AOAIToken.sol         # ERC-20 token with dividends
│   └── scripts/              # Deployment scripts
├── monitoring/               # Prometheus/Grafana configs
├── docker-compose.yml        # Full stack deployment
├── Dockerfile               # Base container image
├── Dockerfile.worker-gpu    # GPU worker image
└── pyproject.toml           # Python package config
```

## 🤝 Contributing

We welcome contributions! This is a community project.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing`)
5. Open a Pull Request

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

**ActuallyOpenAI - Because AI should be Actually Open. Own it. Train it. Earn from it.**
