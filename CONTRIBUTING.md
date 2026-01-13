# Contributing to ActuallyOpenAI

First off, thank you for considering contributing to ActuallyOpenAI! It's people like you that make this decentralized AI platform possible.

## 🌟 Ways to Contribute

### 1. **Contribute Compute Power**
The easiest way to contribute is by running a worker node:
```bash
# Using Docker
docker compose --profile worker-gpu up -d

# Or locally
aoai worker start --wallet YOUR_WALLET_ADDRESS
```
You'll earn AOAI tokens for your contributions!

### 2. **Report Bugs**
Found a bug? Please open an issue with:
- A clear title and description
- Steps to reproduce the issue
- Expected vs actual behavior
- Your environment (OS, Python version, GPU if applicable)

### 3. **Suggest Features**
Have an idea? Open an issue with the `enhancement` label describing:
- The problem you're trying to solve
- Your proposed solution
- Any alternatives you've considered

### 4. **Submit Code**
Ready to code? Follow these steps:

## 🔧 Development Setup

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/actuallyopenai.git
cd actuallyopenai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## 📝 Pull Request Process

1. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. **Make your changes**
   - Write clear, commented code
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed

3. **Run tests**
   ```bash
   # Run all tests
   pytest tests/ -v
   
   # Run with coverage
   pytest tests/ --cov=actuallyopenai
   ```

4. **Commit with a clear message**
   ```bash
   git commit -m "feat: add new verification method"
   # or
   git commit -m "fix: resolve worker heartbeat timeout"
   ```
   
   We follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` new feature
   - `fix:` bug fix
   - `docs:` documentation
   - `test:` adding tests
   - `refactor:` code refactoring

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a Pull Request on GitHub.

## 🎯 Code Style

- **Python**: We use `black` for formatting, `ruff` for linting
- **Type hints**: Always use type hints for function arguments and returns
- **Docstrings**: Use Google-style docstrings
- **Tests**: Aim for >80% code coverage

```python
def calculate_reward(gpu_hours: float, reputation: float) -> int:
    """
    Calculate AOAI token reward for a worker.
    
    Args:
        gpu_hours: Number of GPU hours contributed
        reputation: Worker reputation score (0.0 to 1.0)
    
    Returns:
        Number of AOAI tokens to reward
    
    Example:
        >>> calculate_reward(10.0, 0.95)
        190
    """
    return int(gpu_hours * reputation * BASE_REWARD_RATE)
```

## 🏆 Recognition

All contributors are recognized in our:
- [CONTRIBUTORS.md](CONTRIBUTORS.md) file
- Monthly community updates
- Token airdrops for significant contributions

## 📜 Code of Conduct

### Our Pledge
We pledge to make participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity, level of experience, nationality, personal appearance, race, religion, or sexual identity.

### Our Standards
- Use welcoming and inclusive language
- Be respectful of differing viewpoints
- Accept constructive criticism gracefully
- Focus on what is best for the community

### Enforcement
Instances of unacceptable behavior may be reported to conduct@actuallyopenai.org. All complaints will be reviewed and investigated.

## ❓ Questions?

- Join our [Discord](https://discord.gg/actuallyopenai)
- Check the [Documentation](https://docs.actuallyopenai.org)
- Open a [Discussion](https://github.com/actuallyopenai/actuallyopenai/discussions)

Thank you for being part of the ActuallyOpenAI community! 🚀
