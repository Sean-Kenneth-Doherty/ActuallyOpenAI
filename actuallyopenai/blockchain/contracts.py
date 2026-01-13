"""
AOAI Token Smart Contract - ERC-20 with dividend distribution.
This is a Solidity contract for the ActuallyOpenAI token.
"""

# Solidity source code for AOAI Token
AOAI_TOKEN_CONTRACT = """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Burnable.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";

/**
 * @title ActuallyOpenAI Token (AOAI)
 * @notice ERC-20 token that rewards compute contributors with dividends
 * @dev Includes minting for compute rewards and dividend distribution from API revenue
 */
contract AOAIToken is ERC20, ERC20Burnable, AccessControl, ReentrancyGuard {
    // Roles
    bytes32 public constant MINTER_ROLE = keccak256("MINTER_ROLE");
    bytes32 public constant TREASURY_ROLE = keccak256("TREASURY_ROLE");
    
    // Dividend tracking
    uint256 public totalDividends;
    uint256 public dividendsPerToken;
    mapping(address => uint256) public lastDividendPoints;
    mapping(address => uint256) public pendingDividends;
    
    // Constants
    uint256 private constant PRECISION = 1e18;
    uint256 public constant MAX_SUPPLY = 1_000_000_000 * 1e18; // 1 billion tokens
    
    // Events
    event DividendsDeposited(uint256 amount, uint256 timestamp);
    event DividendsClaimed(address indexed account, uint256 amount);
    event ComputeRewardMinted(address indexed contributor, uint256 amount, string taskId);
    
    constructor() ERC20("ActuallyOpenAI", "AOAI") {
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(MINTER_ROLE, msg.sender);
        _grantRole(TREASURY_ROLE, msg.sender);
    }
    
    /**
     * @notice Mint tokens to reward compute contributors
     * @param to The contributor's wallet address
     * @param amount Amount of tokens to mint
     * @param taskId The training task ID for record-keeping
     */
    function mintComputeReward(
        address to,
        uint256 amount,
        string calldata taskId
    ) external onlyRole(MINTER_ROLE) {
        require(totalSupply() + amount <= MAX_SUPPLY, "Exceeds max supply");
        _updateDividends(to);
        _mint(to, amount);
        emit ComputeRewardMinted(to, amount, taskId);
    }
    
    /**
     * @notice Deposit dividends from API revenue
     * @dev ETH sent to this function is distributed to all token holders
     */
    function depositDividends() external payable onlyRole(TREASURY_ROLE) {
        require(msg.value > 0, "No ETH sent");
        require(totalSupply() > 0, "No tokens in circulation");
        
        dividendsPerToken += (msg.value * PRECISION) / totalSupply();
        totalDividends += msg.value;
        
        emit DividendsDeposited(msg.value, block.timestamp);
    }
    
    /**
     * @notice Claim accumulated dividends
     */
    function claimDividends() external nonReentrant {
        _updateDividends(msg.sender);
        
        uint256 amount = pendingDividends[msg.sender];
        require(amount > 0, "No dividends to claim");
        
        pendingDividends[msg.sender] = 0;
        
        (bool success, ) = payable(msg.sender).call{value: amount}("");
        require(success, "Transfer failed");
        
        emit DividendsClaimed(msg.sender, amount);
    }
    
    /**
     * @notice View pending dividends for an account
     * @param account The account to check
     * @return The amount of pending dividends in wei
     */
    function viewPendingDividends(address account) external view returns (uint256) {
        uint256 newDividends = ((dividendsPerToken - lastDividendPoints[account]) * balanceOf(account)) / PRECISION;
        return pendingDividends[account] + newDividends;
    }
    
    /**
     * @notice Update dividend tracking for an account
     * @param account The account to update
     */
    function _updateDividends(address account) internal {
        if (balanceOf(account) > 0) {
            uint256 newDividends = ((dividendsPerToken - lastDividendPoints[account]) * balanceOf(account)) / PRECISION;
            pendingDividends[account] += newDividends;
        }
        lastDividendPoints[account] = dividendsPerToken;
    }
    
    /**
     * @dev Override transfer to update dividends
     */
    function _beforeTokenTransfer(
        address from,
        address to,
        uint256 amount
    ) internal override {
        if (from != address(0)) {
            _updateDividends(from);
        }
        if (to != address(0)) {
            _updateDividends(to);
        }
        super._beforeTokenTransfer(from, to, amount);
    }
    
    /**
     * @notice Emergency withdrawal for admin
     */
    function emergencyWithdraw() external onlyRole(DEFAULT_ADMIN_ROLE) {
        uint256 balance = address(this).balance;
        (bool success, ) = payable(msg.sender).call{value: balance}("");
        require(success, "Transfer failed");
    }
    
    /**
     * @notice Get contract statistics
     */
    function getStats() external view returns (
        uint256 _totalSupply,
        uint256 _maxSupply,
        uint256 _totalDividends,
        uint256 _dividendsPerToken
    ) {
        return (totalSupply(), MAX_SUPPLY, totalDividends, dividendsPerToken);
    }
}
"""

# Contract ABI for interaction
AOAI_TOKEN_ABI = [
    {
        "inputs": [],
        "stateMutability": "nonpayable",
        "type": "constructor"
    },
    {
        "inputs": [
            {"internalType": "address", "name": "to", "type": "address"},
            {"internalType": "uint256", "name": "amount", "type": "uint256"},
            {"internalType": "string", "name": "taskId", "type": "string"}
        ],
        "name": "mintComputeReward",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "depositDividends",
        "outputs": [],
        "stateMutability": "payable",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "claimDividends",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [{"internalType": "address", "name": "account", "type": "address"}],
        "name": "viewPendingDividends",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"internalType": "address", "name": "account", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "totalSupply",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "getStats",
        "outputs": [
            {"internalType": "uint256", "name": "_totalSupply", "type": "uint256"},
            {"internalType": "uint256", "name": "_maxSupply", "type": "uint256"},
            {"internalType": "uint256", "name": "_totalDividends", "type": "uint256"},
            {"internalType": "uint256", "name": "_dividendsPerToken", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "address", "name": "to", "type": "address"},
            {"internalType": "uint256", "name": "amount", "type": "uint256"}
        ],
        "name": "transfer",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "address", "name": "contributor", "type": "address"},
            {"indexed": False, "internalType": "uint256", "name": "amount", "type": "uint256"},
            {"indexed": False, "internalType": "string", "name": "taskId", "type": "string"}
        ],
        "name": "ComputeRewardMinted",
        "type": "event"
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "address", "name": "account", "type": "address"},
            {"indexed": False, "internalType": "uint256", "name": "amount", "type": "uint256"}
        ],
        "name": "DividendsClaimed",
        "type": "event"
    }
]
