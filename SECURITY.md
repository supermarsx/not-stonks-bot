# Security Policy

## Overview

We take the security of not-stonks-bot seriously. This policy outlines our commitment to security and the process for reporting vulnerabilities.

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

### How to Report

**DO NOT** create public GitHub issues for security vulnerabilities. Instead:

1. **Email**: Send details to [security@not-stonks-bot.com](mailto:security@not-stonks-bot.com)
2. **Subject**: Include "[SECURITY]" in the subject line
3. **Response Time**: We respond within 48 hours

### What to Include

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fixes (if any)
- Your contact information

### Response Process

1. **Acknowledgment** (48 hours): We'll confirm receipt
2. **Initial Assessment** (7 days): We'll review and prioritize
3. **Investigation** (30 days): We'll investigate and develop fixes
4. **Disclosure** (90 days): We'll coordinate responsible disclosure

### What We Won't Do

- Legal action against security researchers acting in good faith
- Request non-disclosure agreements for vulnerability reports
- Require disclosure through channels other than email

### Safe Harbor

We support security research conducted in good faith and will not pursue legal action against researchers who:

- Make a good faith effort to avoid privacy violations
- Do not access or modify user data beyond what's necessary
- Report vulnerabilities promptly
- Allow reasonable time for fixes before public disclosure

## Security Best Practices

### For Users

- **API Keys**: Never commit API keys to version control
- **Environment Variables**: Use secure environment variable management
- **Updates**: Keep your not-stonks-bot installation updated
- **Network Security**: Use HTTPS in production environments

### For Developers

- **Code Review**: All changes require security-focused code review
- **Dependencies**: Regular security audits of dependencies
- **Testing**: Security testing in CI/CD pipeline
- **Documentation**: Security implications documented in code

## Security Measures

### Application Security

- **Input Validation**: All user inputs are validated and sanitized
- **Authentication**: Multi-factor authentication support
- **Authorization**: Role-based access control
- **Encryption**: AES-256 encryption for sensitive data

### Infrastructure Security

- **Network Security**: Firewall and intrusion detection
- **Monitoring**: 24/7 security monitoring
- **Backup**: Encrypted, regular backups
- **Incident Response**: Documented incident response procedures

## Contact Information

- **Security Email**: [security@not-stonks-bot.com](mailto:security@not-stonks-bot.com)
- **General Support**: [support@not-stonks-bot.com](mailto:support@not-stonks-bot.com)
- **PGP Key**: Available upon request

## Recognition

We maintain a security hall of fame for researchers who responsibly disclose vulnerabilities:

- Contributors who report valid security issues
- Researchers who help improve our security posture
- Community members who promote security best practices

Thank you for helping keep not-stonks-bot and its users safe!