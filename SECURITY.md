# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to: **security@actuallyopenai.org**

You should receive a response within 48 hours. If for some reason you do not, please follow up via email to ensure we received your original message.

Please include the following information:
- Type of issue (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
- Full paths of source file(s) related to the issue
- Location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

## Security Measures

ActuallyOpenAI implements several security measures:

### API Security
- JWT-based authentication with short-lived tokens
- API key hashing (SHA-256)
- Rate limiting per user/key
- Input validation on all endpoints

### Worker Security
- Proof-of-Work verification
- Reputation-based trust system
- Gradient verification to detect malicious updates
- Consensus mechanisms for result validation

### Smart Contract Security
- Audited Solidity contracts (when deployed)
- Multi-sig for treasury operations
- Timelock on governance actions

## Bug Bounty Program

We offer rewards for responsible disclosure of security vulnerabilities:

| Severity | Reward |
|----------|--------|
| Critical | Up to 10,000 AOAI |
| High     | Up to 5,000 AOAI  |
| Medium   | Up to 1,000 AOAI  |
| Low      | Up to 250 AOAI    |

Severity is determined based on CVSS score and potential impact.

## Disclosure Policy

- We will acknowledge receipt of your report within 48 hours
- We will provide an estimated timeline for a fix within 7 days
- We will notify you when the vulnerability is fixed
- We will publicly acknowledge your contribution (unless you prefer anonymity)
