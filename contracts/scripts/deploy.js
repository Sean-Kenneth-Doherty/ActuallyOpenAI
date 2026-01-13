const hre = require("hardhat");

async function main() {
  console.log("Deploying ActuallyOpenAI Token (AOAI)...");
  
  const [deployer] = await hre.ethers.getSigners();
  console.log("Deploying with account:", deployer.address);
  console.log("Account balance:", (await deployer.provider.getBalance(deployer.address)).toString());

  // Deploy the token
  const AOAIToken = await hre.ethers.getContractFactory("AOAIToken");
  const token = await AOAIToken.deploy();
  await token.waitForDeployment();

  const tokenAddress = await token.getAddress();
  console.log("AOAI Token deployed to:", tokenAddress);

  // Log initial stats
  const stats = await token.getStats();
  console.log("\nInitial Token Stats:");
  console.log("- Total Supply:", hre.ethers.formatEther(stats[0]), "AOAI");
  console.log("- Max Supply:", hre.ethers.formatEther(stats[1]), "AOAI");
  console.log("- Total Dividends:", hre.ethers.formatEther(stats[2]), "ETH");

  // Save deployment info
  const fs = require("fs");
  const deploymentInfo = {
    network: hre.network.name,
    tokenAddress: tokenAddress,
    deployer: deployer.address,
    deployedAt: new Date().toISOString(),
    blockNumber: await hre.ethers.provider.getBlockNumber()
  };

  fs.writeFileSync(
    `deployments/${hre.network.name}.json`,
    JSON.stringify(deploymentInfo, null, 2)
  );

  console.log("\nDeployment info saved to deployments/" + hre.network.name + ".json");

  // Verify on Etherscan (if not local)
  if (hre.network.name !== "hardhat" && hre.network.name !== "localhost") {
    console.log("\nWaiting for block confirmations...");
    await token.deploymentTransaction().wait(5);
    
    console.log("Verifying contract on Etherscan...");
    try {
      await hre.run("verify:verify", {
        address: tokenAddress,
        constructorArguments: []
      });
      console.log("Contract verified!");
    } catch (error) {
      console.log("Verification failed:", error.message);
    }
  }
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
