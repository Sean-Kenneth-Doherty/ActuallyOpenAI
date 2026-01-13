"""
Token Service - Handles blockchain interactions for AOAI token.
Mints rewards to compute contributors and manages dividend payouts.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, Any, List
from datetime import datetime

from web3 import Web3, AsyncWeb3
from web3.middleware import geth_poa_middleware
from eth_account import Account
from eth_account.signers.local import LocalAccount
import structlog

from actuallyopenai.config import get_settings
from actuallyopenai.blockchain.contracts import AOAI_TOKEN_ABI

logger = structlog.get_logger()


class TokenService:
    """
    Service for interacting with the AOAI token smart contract.
    Handles minting compute rewards and dividend distribution.
    """
    
    def __init__(
        self,
        rpc_url: str,
        contract_address: str,
        treasury_private_key: str
    ):
        self.rpc_url = rpc_url
        self.contract_address = Web3.to_checksum_address(contract_address)
        
        # Initialize Web3
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        
        # Add PoA middleware for testnets like Goerli/Sepolia
        self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        
        # Load treasury account
        self.treasury_account: LocalAccount = Account.from_key(treasury_private_key)
        
        # Initialize contract
        self.contract = self.w3.eth.contract(
            address=self.contract_address,
            abi=AOAI_TOKEN_ABI
        )
        
        # Transaction tracking
        self.pending_transactions: Dict[str, Dict] = {}
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to blockchain."""
        return self.w3.is_connected()
    
    async def get_token_stats(self) -> Dict[str, Any]:
        """Get AOAI token statistics."""
        try:
            stats = self.contract.functions.getStats().call()
            return {
                "total_supply": Web3.from_wei(stats[0], "ether"),
                "max_supply": Web3.from_wei(stats[1], "ether"),
                "total_dividends": Web3.from_wei(stats[2], "ether"),
                "dividends_per_token": Web3.from_wei(stats[3], "ether")
            }
        except Exception as e:
            logger.error("Failed to get token stats", error=str(e))
            return {}
    
    async def get_balance(self, wallet_address: str) -> Decimal:
        """Get AOAI token balance for a wallet."""
        try:
            address = Web3.to_checksum_address(wallet_address)
            balance_wei = self.contract.functions.balanceOf(address).call()
            return Decimal(str(Web3.from_wei(balance_wei, "ether")))
        except Exception as e:
            logger.error("Failed to get balance", wallet=wallet_address, error=str(e))
            return Decimal("0")
    
    async def get_pending_dividends(self, wallet_address: str) -> Decimal:
        """Get pending dividend amount for a wallet."""
        try:
            address = Web3.to_checksum_address(wallet_address)
            pending_wei = self.contract.functions.viewPendingDividends(address).call()
            return Decimal(str(Web3.from_wei(pending_wei, "ether")))
        except Exception as e:
            logger.error("Failed to get pending dividends", wallet=wallet_address, error=str(e))
            return Decimal("0")
    
    async def mint_compute_reward(
        self,
        recipient_wallet: str,
        amount: Decimal,
        task_id: str
    ) -> Optional[str]:
        """
        Mint AOAI tokens as reward for compute contribution.
        
        Args:
            recipient_wallet: The contributor's wallet address
            amount: Amount of AOAI tokens to mint
            task_id: The training task ID for record-keeping
            
        Returns:
            Transaction hash if successful, None otherwise
        """
        try:
            recipient = Web3.to_checksum_address(recipient_wallet)
            amount_wei = Web3.to_wei(float(amount), "ether")
            
            # Build transaction
            nonce = self.w3.eth.get_transaction_count(self.treasury_account.address)
            
            tx = self.contract.functions.mintComputeReward(
                recipient,
                amount_wei,
                task_id
            ).build_transaction({
                "from": self.treasury_account.address,
                "nonce": nonce,
                "gas": 150000,
                "gasPrice": self.w3.eth.gas_price
            })
            
            # Sign and send
            signed_tx = self.w3.eth.account.sign_transaction(
                tx, 
                self.treasury_account.key
            )
            
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            tx_hash_hex = tx_hash.hex()
            
            logger.info(
                "Compute reward minted",
                recipient=recipient_wallet,
                amount=str(amount),
                task_id=task_id,
                tx_hash=tx_hash_hex
            )
            
            # Track pending transaction
            self.pending_transactions[tx_hash_hex] = {
                "type": "mint",
                "recipient": recipient_wallet,
                "amount": str(amount),
                "task_id": task_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return tx_hash_hex
            
        except Exception as e:
            logger.error(
                "Failed to mint compute reward",
                recipient=recipient_wallet,
                amount=str(amount),
                error=str(e)
            )
            return None
    
    async def deposit_dividends(self, amount_eth: Decimal) -> Optional[str]:
        """
        Deposit ETH as dividends to be distributed to all token holders.
        
        Args:
            amount_eth: Amount of ETH to deposit as dividends
            
        Returns:
            Transaction hash if successful, None otherwise
        """
        try:
            amount_wei = Web3.to_wei(float(amount_eth), "ether")
            
            # Check treasury balance
            treasury_balance = self.w3.eth.get_balance(self.treasury_account.address)
            if treasury_balance < amount_wei:
                logger.error(
                    "Insufficient treasury balance for dividend deposit",
                    required=str(amount_eth),
                    available=str(Web3.from_wei(treasury_balance, "ether"))
                )
                return None
            
            # Build transaction
            nonce = self.w3.eth.get_transaction_count(self.treasury_account.address)
            
            tx = self.contract.functions.depositDividends().build_transaction({
                "from": self.treasury_account.address,
                "nonce": nonce,
                "value": amount_wei,
                "gas": 100000,
                "gasPrice": self.w3.eth.gas_price
            })
            
            # Sign and send
            signed_tx = self.w3.eth.account.sign_transaction(
                tx,
                self.treasury_account.key
            )
            
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            tx_hash_hex = tx_hash.hex()
            
            logger.info(
                "Dividends deposited",
                amount_eth=str(amount_eth),
                tx_hash=tx_hash_hex
            )
            
            return tx_hash_hex
            
        except Exception as e:
            logger.error("Failed to deposit dividends", error=str(e))
            return None
    
    async def wait_for_transaction(
        self,
        tx_hash: str,
        timeout: int = 120
    ) -> Optional[Dict]:
        """
        Wait for a transaction to be confirmed.
        
        Args:
            tx_hash: Transaction hash to wait for
            timeout: Maximum time to wait in seconds
            
        Returns:
            Transaction receipt if confirmed, None if timeout
        """
        try:
            receipt = self.w3.eth.wait_for_transaction_receipt(
                tx_hash,
                timeout=timeout
            )
            
            # Clean up pending transactions
            if tx_hash in self.pending_transactions:
                del self.pending_transactions[tx_hash]
            
            return {
                "status": "success" if receipt["status"] == 1 else "failed",
                "block_number": receipt["blockNumber"],
                "gas_used": receipt["gasUsed"],
                "tx_hash": tx_hash
            }
            
        except Exception as e:
            logger.error("Transaction wait failed", tx_hash=tx_hash, error=str(e))
            return None
    
    async def batch_mint_rewards(
        self,
        rewards: List[Dict[str, Any]]
    ) -> List[Optional[str]]:
        """
        Mint rewards for multiple contributors in batch.
        
        Args:
            rewards: List of dicts with 'wallet', 'amount', 'task_id'
            
        Returns:
            List of transaction hashes (None for failed mints)
        """
        results = []
        
        for reward in rewards:
            tx_hash = await self.mint_compute_reward(
                recipient_wallet=reward["wallet"],
                amount=Decimal(str(reward["amount"])),
                task_id=reward["task_id"]
            )
            results.append(tx_hash)
            
            # Small delay to avoid nonce issues
            await asyncio.sleep(0.5)
        
        return results


class PayoutManager:
    """
    Manages periodic payouts to compute contributors.
    Aggregates contributions and triggers blockchain payouts.
    """
    
    def __init__(self, token_service: TokenService):
        self.token_service = token_service
        self.pending_payouts: Dict[str, Decimal] = {}  # wallet -> amount
        self.payout_history: List[Dict] = []
    
    def add_pending_payout(self, wallet_address: str, amount: Decimal):
        """Add to pending payout for a wallet."""
        wallet = wallet_address.lower()
        if wallet not in self.pending_payouts:
            self.pending_payouts[wallet] = Decimal("0")
        self.pending_payouts[wallet] += amount
    
    async def process_payouts(self, min_threshold: Decimal = Decimal("10.0")) -> int:
        """
        Process all pending payouts above the minimum threshold.
        
        Returns:
            Number of payouts processed
        """
        settings = get_settings()
        processed = 0
        
        for wallet, amount in list(self.pending_payouts.items()):
            if amount >= min_threshold:
                tx_hash = await self.token_service.mint_compute_reward(
                    recipient_wallet=wallet,
                    amount=amount,
                    task_id=f"batch-payout-{datetime.utcnow().isoformat()}"
                )
                
                if tx_hash:
                    self.payout_history.append({
                        "wallet": wallet,
                        "amount": str(amount),
                        "tx_hash": tx_hash,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    del self.pending_payouts[wallet]
                    processed += 1
                    
                    logger.info(
                        "Payout processed",
                        wallet=wallet,
                        amount=str(amount),
                        tx_hash=tx_hash
                    )
        
        return processed
    
    def get_pending_payout(self, wallet_address: str) -> Decimal:
        """Get pending payout amount for a wallet."""
        return self.pending_payouts.get(wallet_address.lower(), Decimal("0"))
    
    def get_total_pending(self) -> Decimal:
        """Get total pending payout amount."""
        return sum(self.pending_payouts.values())


# Factory function
def create_token_service() -> TokenService:
    """Create TokenService from settings."""
    settings = get_settings()
    return TokenService(
        rpc_url=settings.blockchain_rpc_url,
        contract_address=settings.token_contract_address,
        treasury_private_key=settings.treasury_private_key
    )
