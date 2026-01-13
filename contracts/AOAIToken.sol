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
 * 
 * Key Features:
 * - Compute contributors earn AOAI tokens for training AI models
 * - API revenue is distributed as ETH dividends to all token holders
 * - Transparent, on-chain tracking of all contributions and rewards
 */
contract AOAIToken is ERC20, ERC20Burnable, AccessControl, ReentrancyGuard {
    // =============================================================================
    // Roles
    // =============================================================================
    
    bytes32 public constant MINTER_ROLE = keccak256("MINTER_ROLE");
    bytes32 public constant TREASURY_ROLE = keccak256("TREASURY_ROLE");
    
    // =============================================================================
    // Dividend Tracking
    // =============================================================================
    
    uint256 public totalDividends;
    uint256 public dividendsPerToken;
    uint256 public totalDividendsClaimed;
    
    mapping(address => uint256) public lastDividendPoints;
    mapping(address => uint256) public pendingDividends;
    mapping(address => uint256) public totalDividendsClaimed_user;
    
    // =============================================================================
    // Compute Contribution Tracking
    // =============================================================================
    
    struct Contribution {
        address contributor;
        uint256 amount;
        string taskId;
        uint256 timestamp;
    }
    
    Contribution[] public contributions;
    mapping(address => uint256) public totalContributed;
    mapping(address => uint256) public contributionCount;
    
    // =============================================================================
    // Constants
    // =============================================================================
    
    uint256 private constant PRECISION = 1e18;
    uint256 public constant MAX_SUPPLY = 1_000_000_000 * 1e18; // 1 billion tokens
    
    // =============================================================================
    // Events
    // =============================================================================
    
    event DividendsDeposited(
        uint256 amount, 
        uint256 newDividendsPerToken,
        uint256 timestamp
    );
    
    event DividendsClaimed(
        address indexed account, 
        uint256 amount,
        uint256 timestamp
    );
    
    event ComputeRewardMinted(
        address indexed contributor, 
        uint256 amount, 
        string taskId,
        uint256 timestamp
    );
    
    event ContributorRegistered(
        address indexed contributor,
        uint256 timestamp
    );
    
    // =============================================================================
    // Constructor
    // =============================================================================
    
    constructor() ERC20("ActuallyOpenAI", "AOAI") {
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(MINTER_ROLE, msg.sender);
        _grantRole(TREASURY_ROLE, msg.sender);
    }
    
    // =============================================================================
    // Compute Reward Functions
    // =============================================================================
    
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
        require(to != address(0), "Cannot mint to zero address");
        require(amount > 0, "Amount must be greater than 0");
        require(totalSupply() + amount <= MAX_SUPPLY, "Exceeds max supply");
        
        // Update dividends before mint
        _updateDividends(to);
        
        // Mint tokens
        _mint(to, amount);
        
        // Record contribution
        contributions.push(Contribution({
            contributor: to,
            amount: amount,
            taskId: taskId,
            timestamp: block.timestamp
        }));
        
        totalContributed[to] += amount;
        contributionCount[to]++;
        
        emit ComputeRewardMinted(to, amount, taskId, block.timestamp);
    }
    
    /**
     * @notice Batch mint rewards to multiple contributors
     * @param recipients Array of contributor addresses
     * @param amounts Array of amounts to mint
     * @param taskIds Array of task IDs
     */
    function batchMintComputeReward(
        address[] calldata recipients,
        uint256[] calldata amounts,
        string[] calldata taskIds
    ) external onlyRole(MINTER_ROLE) {
        require(
            recipients.length == amounts.length && 
            amounts.length == taskIds.length,
            "Array lengths must match"
        );
        
        uint256 totalAmount = 0;
        for (uint256 i = 0; i < amounts.length; i++) {
            totalAmount += amounts[i];
        }
        require(totalSupply() + totalAmount <= MAX_SUPPLY, "Exceeds max supply");
        
        for (uint256 i = 0; i < recipients.length; i++) {
            if (recipients[i] != address(0) && amounts[i] > 0) {
                _updateDividends(recipients[i]);
                _mint(recipients[i], amounts[i]);
                
                contributions.push(Contribution({
                    contributor: recipients[i],
                    amount: amounts[i],
                    taskId: taskIds[i],
                    timestamp: block.timestamp
                }));
                
                totalContributed[recipients[i]] += amounts[i];
                contributionCount[recipients[i]]++;
                
                emit ComputeRewardMinted(
                    recipients[i], 
                    amounts[i], 
                    taskIds[i], 
                    block.timestamp
                );
            }
        }
    }
    
    // =============================================================================
    // Dividend Functions
    // =============================================================================
    
    /**
     * @notice Deposit dividends from API revenue
     * @dev ETH sent to this function is distributed to all token holders
     */
    function depositDividends() external payable onlyRole(TREASURY_ROLE) {
        require(msg.value > 0, "No ETH sent");
        require(totalSupply() > 0, "No tokens in circulation");
        
        uint256 newDividendsPerToken = (msg.value * PRECISION) / totalSupply();
        dividendsPerToken += newDividendsPerToken;
        totalDividends += msg.value;
        
        emit DividendsDeposited(msg.value, dividendsPerToken, block.timestamp);
    }
    
    /**
     * @notice Claim accumulated dividends
     */
    function claimDividends() external nonReentrant {
        _updateDividends(msg.sender);
        
        uint256 amount = pendingDividends[msg.sender];
        require(amount > 0, "No dividends to claim");
        
        pendingDividends[msg.sender] = 0;
        totalDividendsClaimed += amount;
        totalDividendsClaimed_user[msg.sender] += amount;
        
        (bool success, ) = payable(msg.sender).call{value: amount}("");
        require(success, "ETH transfer failed");
        
        emit DividendsClaimed(msg.sender, amount, block.timestamp);
    }
    
    /**
     * @notice View pending dividends for an account
     * @param account The account to check
     * @return The amount of pending dividends in wei
     */
    function viewPendingDividends(address account) external view returns (uint256) {
        uint256 newDividends = 0;
        if (balanceOf(account) > 0) {
            newDividends = ((dividendsPerToken - lastDividendPoints[account]) * balanceOf(account)) / PRECISION;
        }
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
    
    // =============================================================================
    // Transfer Overrides
    // =============================================================================
    
    /**
     * @dev Override _beforeTokenTransfer to update dividends on transfers
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
    
    // =============================================================================
    // View Functions
    // =============================================================================
    
    /**
     * @notice Get contract statistics
     */
    function getStats() external view returns (
        uint256 _totalSupply,
        uint256 _maxSupply,
        uint256 _totalDividends,
        uint256 _totalDividendsClaimed,
        uint256 _dividendsPerToken,
        uint256 _totalContributions
    ) {
        return (
            totalSupply(), 
            MAX_SUPPLY, 
            totalDividends,
            totalDividendsClaimed,
            dividendsPerToken,
            contributions.length
        );
    }
    
    /**
     * @notice Get contributor statistics
     * @param account The account to check
     */
    function getContributorStats(address account) external view returns (
        uint256 _balance,
        uint256 _totalContributed,
        uint256 _contributionCount,
        uint256 _pendingDividends,
        uint256 _totalDividendsClaimed
    ) {
        uint256 pending = pendingDividends[account];
        if (balanceOf(account) > 0) {
            pending += ((dividendsPerToken - lastDividendPoints[account]) * balanceOf(account)) / PRECISION;
        }
        
        return (
            balanceOf(account),
            totalContributed[account],
            contributionCount[account],
            pending,
            totalDividendsClaimed_user[account]
        );
    }
    
    /**
     * @notice Get recent contributions
     * @param count Number of recent contributions to return
     */
    function getRecentContributions(uint256 count) external view returns (
        address[] memory contributors,
        uint256[] memory amounts,
        uint256[] memory timestamps
    ) {
        uint256 total = contributions.length;
        uint256 returnCount = count > total ? total : count;
        
        contributors = new address[](returnCount);
        amounts = new uint256[](returnCount);
        timestamps = new uint256[](returnCount);
        
        for (uint256 i = 0; i < returnCount; i++) {
            Contribution storage c = contributions[total - 1 - i];
            contributors[i] = c.contributor;
            amounts[i] = c.amount;
            timestamps[i] = c.timestamp;
        }
        
        return (contributors, amounts, timestamps);
    }
    
    // =============================================================================
    // Admin Functions
    // =============================================================================
    
    /**
     * @notice Emergency withdrawal for admin
     */
    function emergencyWithdraw() external onlyRole(DEFAULT_ADMIN_ROLE) {
        uint256 balance = address(this).balance;
        require(balance > 0, "No ETH to withdraw");
        (bool success, ) = payable(msg.sender).call{value: balance}("");
        require(success, "Transfer failed");
    }
    
    /**
     * @notice Receive ETH directly (treated as dividend deposit)
     */
    receive() external payable {
        if (msg.value > 0 && totalSupply() > 0) {
            uint256 newDividendsPerToken = (msg.value * PRECISION) / totalSupply();
            dividendsPerToken += newDividendsPerToken;
            totalDividends += msg.value;
            emit DividendsDeposited(msg.value, dividendsPerToken, block.timestamp);
        }
    }
}
