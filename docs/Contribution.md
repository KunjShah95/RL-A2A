
# Contributing to the Agent-to-Agent Protocol

Thank you for your interest in contributing to the Agent-to-Agent Protocol! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Process](#development-process)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation Guidelines](#documentation-guidelines)
- [Issue Reporting](#issue-reporting)
- [Feature Requests](#feature-requests)
- [Community](#community)

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

### Prerequisites

- Python 3.8+
- Git
- Basic understanding of FastAPI, WebSockets, and reinforcement learning concepts

### Setting Up Development Environment

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/AGENT-TO-AGENT-PROTOCOL-.git
   cd AGENT-TO-AGENT-PROTOCOL-
   ```
3. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # If available
   ```

## Development Process

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bugfix-name
   ```
2. Make your changes
3. Run tests to ensure your changes don't break existing functionality
4. Update documentation as needed
5. Commit your changes with clear, descriptive commit messages
6. Push your branch to your fork
7. Submit a pull request

## Pull Request Process

1. Ensure your PR includes tests for new functionality
2. Update the README.md or documentation with details of changes if applicable
3. The PR should work on the main development branch
4. Include a description of the changes and why they should be included
5. Link any related issues using GitHub's keywords (e.g., "Fixes #123")
6. Wait for review from maintainers

## Coding Standards

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines
- Use meaningful variable and function names
- Include docstrings for all functions, classes, and modules
- Keep functions focused on a single responsibility
- Use type hints where appropriate
- Format your code with [Black](https://black.readthedocs.io/)
- Use [isort](https://pycqa.github.io/isort/) to sort imports

Example docstring format:

```python
def function_name(param1: type, param2: type) -> return_type:
    """
    Short description of function.

    Longer description explaining the function's purpose,
    behavior, and any important details.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ExceptionType: When and why this exception is raised
    """
```

## Testing Guidelines

- Write unit tests for all new functionality
- Ensure all tests pass before submitting a PR
- Aim for high test coverage
- Use pytest for testing
- Place tests in the `tests/` directory with a structure mirroring the main code

Running tests:
```bash
pytest
```

## Documentation Guidelines

- Keep documentation up to date with code changes
- Use clear, concise language
- Include examples where appropriate
- Follow Markdown best practices
- Document public APIs thoroughly
- Update the changelog for notable changes

## Issue Reporting

When reporting issues, please include:

1. A clear, descriptive title
2. Steps to reproduce the issue
3. Expected behavior
4. Actual behavior
5. Environment details (OS, Python version, etc.)
6. Any relevant logs or error messages

## Feature Requests

Feature requests are welcome! Please provide:

1. A clear description of the feature
2. The motivation for the feature
3. How it would benefit the project
4. Any implementation ideas you have

## Community

- Join discussions in the GitHub Issues and Pull Requests
- Be respectful and constructive in all communications
- Help others who have questions
- Share your use cases and experiences

Thank you for contributing to the Agent-to-Agent Protocol!
