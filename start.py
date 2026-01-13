#!/usr/bin/env python3
"""
ActuallyOpenAI - Quick Start Script

This script helps you get started with ActuallyOpenAI quickly.

Usage:
    python start.py api        # Start the API server
    python start.py train      # Train the model
    python start.py worker     # Start a worker node
    python start.py demo       # Run the demo
    python start.py test       # Run tests
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

# Add project to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def start_api(host: str = "0.0.0.0", port: int = 8000):
    """Start the ActuallyOpenAI API server."""
    print("=" * 60)
    print("  Starting ActuallyOpenAI API Server")
    print("=" * 60)
    print(f"\n  URL: http://{host}:{port}")
    print(f"  Docs: http://{host}:{port}/docs")
    print(f"  Demo API Key: aoai-demo-key-123456789")
    print("\n  Press Ctrl+C to stop\n")
    print("=" * 60)
    
    os.system(f"python -m uvicorn actuallyopenai.api.production_api:app --host {host} --port {port}")


def start_training(epochs: int = 10, batch_size: int = 16):
    """Train the AOAI model."""
    print("=" * 60)
    print("  Starting ActuallyOpenAI Model Training")
    print("=" * 60)
    
    # Create training data if not exists
    data_path = PROJECT_ROOT / "data" / "train.txt"
    if not data_path.exists():
        print("\nCreating training data...")
        os.system("python scripts/prepare_data.py")
    
    # Start training
    cmd = f"python scripts/train_model.py --data data/train.txt --epochs {epochs} --batch-size {batch_size}"
    os.system(cmd)


def start_worker(wallet: str = None):
    """Start a worker node to contribute compute."""
    print("=" * 60)
    print("  Starting ActuallyOpenAI Worker Node")
    print("=" * 60)
    
    if wallet:
        print(f"\n  Wallet: {wallet}")
    else:
        print("\n  No wallet specified - using demo mode")
        wallet = "0x0000000000000000000000000000000000000000"
    
    print("  Connecting to orchestrator...")
    
    # For now, run the orchestrator locally
    os.system("python -m uvicorn actuallyopenai.orchestrator.main:app --host 0.0.0.0 --port 8001")


def run_demo():
    """Run the full system demonstration."""
    print("=" * 60)
    print("  Running ActuallyOpenAI Demo")
    print("=" * 60)
    
    os.system("python scripts/full_demo.py")


def run_tests():
    """Run the test suite."""
    print("=" * 60)
    print("  Running ActuallyOpenAI Tests")
    print("=" * 60)
    
    os.system("python -m pytest tests/ -v")


def show_info():
    """Show project information."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     █████╗  ██████╗ ████████╗██╗   ██╗ █████╗ ██╗     ██╗    ║
║    ██╔══██╗██╔════╝ ╚══██╔══╝██║   ██║██╔══██╗██║     ██║    ║
║    ███████║██║         ██║   ██║   ██║███████║██║     ██║    ║
║    ██╔══██║██║         ██║   ██║   ██║██╔══██║██║     ██║    ║
║    ██║  ██║╚██████╗    ██║   ╚██████╔╝██║  ██║███████╗██║    ║
║    ╚═╝  ╚═╝ ╚═════╝    ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝    ║
║                                                              ║
║              █████╗ ██████╗ ███████╗███╗   ██╗               ║
║             ██╔══██╗██╔══██╗██╔════╝████╗  ██║               ║
║             ██║  ██║██████╔╝█████╗  ██╔██╗ ██║               ║
║             ██║  ██║██╔═══╝ ██╔══╝  ██║╚██╗██║               ║
║             ╚█████╔╝██║     ███████╗██║ ╚████║               ║
║              ╚════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝               ║
║                                                              ║
║                   AI for Everyone                            ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Commands:                                                   ║
║    python start.py api      - Start the API server           ║
║    python start.py train    - Train the model                ║
║    python start.py worker   - Start a worker node            ║
║    python start.py demo     - Run the demo                   ║
║    python start.py test     - Run tests                      ║
║                                                              ║
║  Quick Start:                                                ║
║    1. python start.py api   - Start the server               ║
║    2. Visit http://localhost:8000/docs                       ║
║    3. Use API key: aoai-demo-key-123456789                   ║
║                                                              ║
║  GitHub: https://github.com/actuallyopenai/actuallyopenai    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


def main():
    parser = argparse.ArgumentParser(
        description="ActuallyOpenAI - Quick Start",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python start.py api               # Start API on default port
    python start.py api --port 9000   # Start API on port 9000
    python start.py train --epochs 20 # Train for 20 epochs
    python start.py worker --wallet YOUR_WALLET
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # API command
    api_parser = subparsers.add_parser("api", help="Start the API server")
    api_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    api_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    train_parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    
    # Worker command
    worker_parser = subparsers.add_parser("worker", help="Start a worker node")
    worker_parser.add_argument("--wallet", help="Your wallet address for rewards")
    
    # Demo command
    subparsers.add_parser("demo", help="Run the demo")
    
    # Test command
    subparsers.add_parser("test", help="Run tests")
    
    # Info command
    subparsers.add_parser("info", help="Show project information")
    
    args = parser.parse_args()
    
    if args.command == "api":
        start_api(args.host, args.port)
    elif args.command == "train":
        start_training(args.epochs, args.batch_size)
    elif args.command == "worker":
        start_worker(args.wallet)
    elif args.command == "demo":
        run_demo()
    elif args.command == "test":
        run_tests()
    elif args.command == "info":
        show_info()
    else:
        show_info()
        parser.print_help()


if __name__ == "__main__":
    main()
