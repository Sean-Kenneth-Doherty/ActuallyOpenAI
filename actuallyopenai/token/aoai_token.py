"""
AOAI Token System
=================
The native token of ActuallyOpenAI network.

Contribute compute → Earn AOAI tokens
API revenue → Distributed to token holders

Like Bitcoin mining, but you're training AI instead of solving useless puzzles.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import threading
import logging
from pathlib import Path

logger = logging.getLogger("AOAI-Token")


class TransactionType(Enum):
    """Types of transactions"""
    REWARD = "reward"           # Compute contribution reward
    TRANSFER = "transfer"       # Token transfer between wallets
    DIVIDEND = "dividend"       # API revenue distribution
    STAKE = "stake"            # Staking tokens
    UNSTAKE = "unstake"        # Unstaking tokens
    BURN = "burn"              # Token burn (deflationary)


@dataclass
class Transaction:
    """A single transaction in the ledger"""
    tx_type: TransactionType
    sender: str              # Wallet address or "NETWORK" for rewards
    receiver: str            # Wallet address
    amount: float            # AOAI tokens
    timestamp: float = field(default_factory=time.time)
    data: dict = field(default_factory=dict)  # Additional data
    signature: str = ""      # Cryptographic signature
    
    @property
    def tx_id(self) -> str:
        """Transaction ID (hash)"""
        content = f"{self.tx_type.value}{self.sender}{self.receiver}{self.amount}{self.timestamp}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def to_dict(self) -> dict:
        return {
            "tx_id": self.tx_id,
            "type": self.tx_type.value,
            "sender": self.sender,
            "receiver": self.receiver,
            "amount": self.amount,
            "timestamp": self.timestamp,
            "data": self.data,
            "signature": self.signature
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Transaction':
        return cls(
            tx_type=TransactionType(data["type"]),
            sender=data["sender"],
            receiver=data["receiver"],
            amount=data["amount"],
            timestamp=data["timestamp"],
            data=data.get("data", {}),
            signature=data.get("signature", "")
        )


@dataclass
class Block:
    """A block in the ledger"""
    index: int
    transactions: List[Transaction]
    previous_hash: str
    timestamp: float = field(default_factory=time.time)
    nonce: int = 0
    
    @property
    def hash(self) -> str:
        """Block hash"""
        content = f"{self.index}{self.previous_hash}{self.timestamp}{self.nonce}"
        content += "".join(tx.tx_id for tx in self.transactions)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "transactions": [tx.to_dict() for tx in self.transactions],
            "previous_hash": self.previous_hash,
            "timestamp": self.timestamp,
            "nonce": self.nonce,
            "hash": self.hash
        }


class AOAIWallet:
    """
    AOAI Token Wallet
    
    Each node has a wallet. Contribute compute, earn tokens.
    """
    
    def __init__(self, private_key: Optional[str] = None):
        # Generate or use provided private key
        if private_key:
            self.private_key = private_key
        else:
            self.private_key = hashlib.sha256(
                f"{time.time()}{id(self)}".encode()
            ).hexdigest()
        
        # Derive public address from private key
        self.address = self._derive_address()
        
        logger.info(f"💰 Wallet created: {self.address[:16]}...")
    
    def _derive_address(self) -> str:
        """Derive public address from private key"""
        # Simplified - real implementation would use elliptic curves
        public = hashlib.sha256(self.private_key.encode()).hexdigest()
        return "aoai_" + public[:40]
    
    def sign(self, message: str) -> str:
        """Sign a message with private key"""
        # Simplified - real implementation would use ECDSA
        signature_content = f"{self.private_key}{message}"
        return hashlib.sha256(signature_content.encode()).hexdigest()
    
    def verify(self, message: str, signature: str, address: str) -> bool:
        """Verify a signature (simplified)"""
        # In real implementation, this would verify without private key
        return True  # Simplified for demo
    
    def export(self) -> dict:
        """Export wallet (WARNING: includes private key)"""
        return {
            "address": self.address,
            "private_key": self.private_key
        }
    
    @classmethod
    def from_file(cls, path: str) -> 'AOAIWallet':
        """Load wallet from file"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(private_key=data["private_key"])
    
    def save(self, path: str):
        """Save wallet to file"""
        with open(path, 'w') as f:
            json.dump(self.export(), f)
        logger.info(f"💾 Wallet saved to {path}")


class AOAILedger:
    """
    Decentralized ledger for AOAI tokens
    
    This is a simplified blockchain. In production, this would
    integrate with Ethereum, Solana, or a custom L2.
    """
    
    # Token economics
    TOTAL_SUPPLY = 1_000_000_000  # 1 billion tokens
    INITIAL_REWARD = 100          # Tokens per compute unit
    HALVING_INTERVAL = 100_000    # Halving every 100k blocks
    MIN_REWARD = 0.001            # Minimum reward
    
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir or "./aoai_ledger")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Blockchain
        self.chain: List[Block] = []
        self.pending_transactions: List[Transaction] = []
        
        # State
        self.balances: Dict[str, float] = {}
        self.staked: Dict[str, float] = {}
        self.total_minted: float = 0
        
        # Load or create genesis
        self._load_or_create_genesis()
        
        # Lock for thread safety
        self._lock = threading.Lock()
        
        logger.info(f"📒 Ledger initialized with {len(self.chain)} blocks")
    
    def _load_or_create_genesis(self):
        """Load existing chain or create genesis block"""
        chain_file = self.data_dir / "chain.json"
        
        if chain_file.exists():
            with open(chain_file, 'r') as f:
                data = json.load(f)
                for block_data in data["chain"]:
                    transactions = [
                        Transaction.from_dict(tx)
                        for tx in block_data["transactions"]
                    ]
                    block = Block(
                        index=block_data["index"],
                        transactions=transactions,
                        previous_hash=block_data["previous_hash"],
                        timestamp=block_data["timestamp"],
                        nonce=block_data["nonce"]
                    )
                    self.chain.append(block)
                self.balances = data.get("balances", {})
                self.total_minted = data.get("total_minted", 0)
        else:
            # Create genesis block
            genesis = Block(
                index=0,
                transactions=[],
                previous_hash="0" * 64,
                timestamp=time.time()
            )
            self.chain.append(genesis)
            self._save()
    
    def _save(self):
        """Save chain to disk"""
        chain_file = self.data_dir / "chain.json"
        data = {
            "chain": [block.to_dict() for block in self.chain],
            "balances": self.balances,
            "total_minted": self.total_minted
        }
        with open(chain_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_balance(self, address: str) -> float:
        """Get token balance for address"""
        return self.balances.get(address, 0.0)
    
    def get_staked(self, address: str) -> float:
        """Get staked tokens for address"""
        return self.staked.get(address, 0.0)
    
    def get_current_reward(self) -> float:
        """Calculate current block reward (halving mechanism)"""
        halvings = len(self.chain) // self.HALVING_INTERVAL
        reward = self.INITIAL_REWARD / (2 ** halvings)
        return max(reward, self.MIN_REWARD)
    
    def create_reward_transaction(
        self,
        receiver: str,
        compute_units: float,
        work_proof: str
    ) -> Transaction:
        """
        Create a reward transaction for compute contribution
        
        This is how nodes earn AOAI tokens by training the AI.
        """
        reward = self.get_current_reward() * compute_units
        
        # Check supply cap
        if self.total_minted + reward > self.TOTAL_SUPPLY:
            reward = max(0, self.TOTAL_SUPPLY - self.total_minted)
        
        tx = Transaction(
            tx_type=TransactionType.REWARD,
            sender="NETWORK",
            receiver=receiver,
            amount=reward,
            data={
                "compute_units": compute_units,
                "work_proof": work_proof,
                "block_reward": self.get_current_reward()
            }
        )
        
        return tx
    
    def create_transfer_transaction(
        self,
        wallet: AOAIWallet,
        receiver: str,
        amount: float
    ) -> Transaction:
        """Create a transfer transaction"""
        if self.get_balance(wallet.address) < amount:
            raise ValueError(f"Insufficient balance: {self.get_balance(wallet.address)} < {amount}")
        
        tx = Transaction(
            tx_type=TransactionType.TRANSFER,
            sender=wallet.address,
            receiver=receiver,
            amount=amount
        )
        
        # Sign transaction
        tx.signature = wallet.sign(tx.tx_id)
        
        return tx
    
    def create_stake_transaction(
        self,
        wallet: AOAIWallet,
        amount: float
    ) -> Transaction:
        """Stake tokens for governance and increased rewards"""
        if self.get_balance(wallet.address) < amount:
            raise ValueError("Insufficient balance for staking")
        
        tx = Transaction(
            tx_type=TransactionType.STAKE,
            sender=wallet.address,
            receiver="STAKING_CONTRACT",
            amount=amount
        )
        tx.signature = wallet.sign(tx.tx_id)
        
        return tx
    
    def submit_transaction(self, tx: Transaction) -> str:
        """Submit transaction to pending pool"""
        with self._lock:
            # Validate transaction
            if tx.tx_type == TransactionType.TRANSFER:
                if self.get_balance(tx.sender) < tx.amount:
                    raise ValueError("Insufficient balance")
            
            self.pending_transactions.append(tx)
            logger.info(f"📝 Transaction submitted: {tx.tx_id[:16]}...")
            
            return tx.tx_id
    
    def mine_block(self, miner_address: str) -> Optional[Block]:
        """Mine a new block with pending transactions"""
        with self._lock:
            if not self.pending_transactions:
                return None
            
            # Create new block
            block = Block(
                index=len(self.chain),
                transactions=self.pending_transactions.copy(),
                previous_hash=self.chain[-1].hash
            )
            
            # Simple proof of work (difficulty = 2 leading zeros)
            target = "00"
            while not block.hash.startswith(target):
                block.nonce += 1
            
            # Apply transactions to state
            for tx in block.transactions:
                self._apply_transaction(tx)
            
            # Add block to chain
            self.chain.append(block)
            self.pending_transactions.clear()
            
            # Save to disk
            self._save()
            
            logger.info(f"⛏️ Block mined: #{block.index} ({len(block.transactions)} txs)")
            
            return block
    
    def _apply_transaction(self, tx: Transaction):
        """Apply transaction to state"""
        if tx.tx_type == TransactionType.REWARD:
            self.balances[tx.receiver] = self.balances.get(tx.receiver, 0) + tx.amount
            self.total_minted += tx.amount
            
        elif tx.tx_type == TransactionType.TRANSFER:
            self.balances[tx.sender] = self.balances.get(tx.sender, 0) - tx.amount
            self.balances[tx.receiver] = self.balances.get(tx.receiver, 0) + tx.amount
            
        elif tx.tx_type == TransactionType.STAKE:
            self.balances[tx.sender] = self.balances.get(tx.sender, 0) - tx.amount
            self.staked[tx.sender] = self.staked.get(tx.sender, 0) + tx.amount
            
        elif tx.tx_type == TransactionType.UNSTAKE:
            self.staked[tx.sender] = self.staked.get(tx.sender, 0) - tx.amount
            self.balances[tx.sender] = self.balances.get(tx.sender, 0) + tx.amount
    
    def distribute_dividends(self, total_revenue: float):
        """
        Distribute API revenue to token holders
        
        This is what makes AOAI special - contribute compute,
        earn a share of ALL future revenue. Forever.
        """
        if self.total_minted == 0:
            return
        
        transactions = []
        
        for address, balance in self.balances.items():
            # Calculate share based on token holdings
            share = balance / self.total_minted
            dividend = total_revenue * share
            
            if dividend > 0.001:  # Minimum dividend
                tx = Transaction(
                    tx_type=TransactionType.DIVIDEND,
                    sender="REVENUE_POOL",
                    receiver=address,
                    amount=dividend,
                    data={"revenue": total_revenue, "share": share}
                )
                transactions.append(tx)
        
        # Submit all dividend transactions
        for tx in transactions:
            self.submit_transaction(tx)
        
        logger.info(f"💸 Dividends distributed: ${total_revenue:.2f} to {len(transactions)} holders")
    
    def get_stats(self) -> dict:
        """Get ledger statistics"""
        return {
            "blocks": len(self.chain),
            "total_minted": self.total_minted,
            "remaining_supply": self.TOTAL_SUPPLY - self.total_minted,
            "current_reward": self.get_current_reward(),
            "unique_holders": len(self.balances),
            "total_staked": sum(self.staked.values()),
            "pending_transactions": len(self.pending_transactions)
        }
    
    def get_rich_list(self, limit: int = 10) -> List[Tuple[str, float]]:
        """Get top token holders"""
        sorted_balances = sorted(
            self.balances.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_balances[:limit]


class ComputeRewarder:
    """
    Rewards nodes for compute contributions
    
    Tracks work done, validates proofs, and issues AOAI tokens.
    """
    
    def __init__(self, ledger: AOAILedger):
        self.ledger = ledger
        self.work_history: Dict[str, List[dict]] = {}
    
    def record_work(
        self,
        worker_address: str,
        work_type: str,
        compute_units: float,
        proof: dict
    ) -> float:
        """
        Record work and issue reward
        
        Work types:
        - "training": Training gradient computation
        - "inference": Running inference
        - "validation": Validating other workers
        """
        # Generate work proof hash
        proof_hash = hashlib.sha256(
            json.dumps(proof, sort_keys=True).encode()
        ).hexdigest()
        
        # Create reward transaction
        tx = self.ledger.create_reward_transaction(
            receiver=worker_address,
            compute_units=compute_units,
            work_proof=proof_hash
        )
        
        # Submit to ledger
        self.ledger.submit_transaction(tx)
        
        # Record in history
        if worker_address not in self.work_history:
            self.work_history[worker_address] = []
        
        self.work_history[worker_address].append({
            "type": work_type,
            "compute_units": compute_units,
            "reward": tx.amount,
            "timestamp": time.time(),
            "proof": proof_hash
        })
        
        logger.info(f"⚡ Work recorded: {compute_units:.2f} units -> {tx.amount:.4f} AOAI")
        
        return tx.amount
    
    def get_worker_stats(self, address: str) -> dict:
        """Get statistics for a worker"""
        history = self.work_history.get(address, [])
        
        return {
            "total_work": len(history),
            "total_compute_units": sum(w["compute_units"] for w in history),
            "total_earned": sum(w["reward"] for w in history),
            "balance": self.ledger.get_balance(address),
            "staked": self.ledger.get_staked(address)
        }


# CLI
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="AOAI Token System")
    parser.add_argument("command", choices=[
        "wallet", "balance", "transfer", "stake", "stats", "rich-list"
    ])
    parser.add_argument("--address", help="Wallet address")
    parser.add_argument("--to", help="Recipient address")
    parser.add_argument("--amount", type=float, help="Amount of AOAI")
    parser.add_argument("--wallet-file", default="wallet.json", help="Wallet file path")
    args = parser.parse_args()
    
    ledger = AOAILedger()
    
    if args.command == "wallet":
        # Create or load wallet
        wallet_path = Path(args.wallet_file)
        if wallet_path.exists():
            wallet = AOAIWallet.from_file(str(wallet_path))
            print(f"✅ Wallet loaded: {wallet.address}")
        else:
            wallet = AOAIWallet()
            wallet.save(str(wallet_path))
            print(f"✅ New wallet created: {wallet.address}")
        print(f"   Balance: {ledger.get_balance(wallet.address):.4f} AOAI")
        
    elif args.command == "balance":
        address = args.address
        if not address and Path(args.wallet_file).exists():
            wallet = AOAIWallet.from_file(args.wallet_file)
            address = wallet.address
        if address:
            balance = ledger.get_balance(address)
            staked = ledger.get_staked(address)
            print(f"\n💰 Balance for {address[:20]}...")
            print(f"   Available: {balance:.4f} AOAI")
            print(f"   Staked: {staked:.4f} AOAI")
            print(f"   Total: {balance + staked:.4f} AOAI")
        else:
            print("❌ Provide --address or create a wallet first")
            
    elif args.command == "transfer":
        if not args.to or not args.amount:
            print("❌ --to and --amount required")
            return
        wallet = AOAIWallet.from_file(args.wallet_file)
        tx = ledger.create_transfer_transaction(wallet, args.to, args.amount)
        ledger.submit_transaction(tx)
        ledger.mine_block(wallet.address)
        print(f"✅ Transferred {args.amount:.4f} AOAI to {args.to[:20]}...")
        
    elif args.command == "stake":
        if not args.amount:
            print("❌ --amount required")
            return
        wallet = AOAIWallet.from_file(args.wallet_file)
        tx = ledger.create_stake_transaction(wallet, args.amount)
        ledger.submit_transaction(tx)
        ledger.mine_block(wallet.address)
        print(f"✅ Staked {args.amount:.4f} AOAI")
        
    elif args.command == "stats":
        stats = ledger.get_stats()
        print("\n📊 AOAI Token Statistics")
        print("=" * 40)
        print(f"   Blocks: {stats['blocks']}")
        print(f"   Total Minted: {stats['total_minted']:,.2f} AOAI")
        print(f"   Remaining Supply: {stats['remaining_supply']:,.2f} AOAI")
        print(f"   Current Block Reward: {stats['current_reward']:.4f} AOAI")
        print(f"   Unique Holders: {stats['unique_holders']}")
        print(f"   Total Staked: {stats['total_staked']:,.2f} AOAI")
        
    elif args.command == "rich-list":
        rich = ledger.get_rich_list()
        print("\n🏆 AOAI Rich List (Top 10)")
        print("=" * 50)
        for i, (addr, balance) in enumerate(rich, 1):
            print(f"   {i}. {addr[:20]}... : {balance:,.4f} AOAI")


if __name__ == "__main__":
    main()
