# Contributing to NetPulse

Thank you for your interest in contributing to NetPulse! This document provides guidelines and instructions for contributing.

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Keep discussions professional and on-topic

## How to Contribute

### Reporting Bugs

1. Check existing issues first
2. Create a new issue with:
   - Clear description
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version, etc.)

### Suggesting Features

1. Open an issue describing the feature
2. Explain the use case and benefits
3. Discuss implementation approach if you have ideas

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Write/update tests
5. Ensure all tests pass (`pytest tests/ -v`)
6. Update documentation if needed
7. Submit a pull request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/netpulse.git
cd netpulse

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=netpulse --cov-report=html

# Run specific test category
pytest tests/test_features.py -v
```

## Code Style

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write docstrings for public functions and classes
- Keep functions focused and modular

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
