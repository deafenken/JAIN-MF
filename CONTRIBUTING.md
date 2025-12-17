# Contributing to JAIN-MF

Thank you for your interest in contributing to JAIN-MF! This document provides guidelines for contributors.

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Follow the coding standards

## How to Contribute

### 1. Fork the Repository

Click the "Fork" button on the GitHub repository page.

### 2. Clone Your Fork

```bash
git clone https://github.com/yourusername/JAIN-MF.git
cd JAIN-MF
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 4. Make Your Changes

### 5. Commit Your Changes

```bash
git add .
git commit -m "Add your feature description"
```

### 6. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 7. Create a Pull Request

- Go to your fork on GitHub
- Click "New Pull Request"
- Fill out the pull request template
- Submit the pull request

## Coding Standards

### Python Code Style

We use the following tools to maintain code quality:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting

Install dev dependencies:

```bash
pip install -e .[dev]
```

Format your code:

```bash
black .
isort .
flake8 .
```

### Naming Conventions

- Classes: `PascalCase`
- Functions and variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private methods: `_leading_underscore`

### Documentation

- Add docstrings to all public functions and classes
- Use Google-style docstrings
- Include type hints

Example:

```python
def process_frames(
    frames: torch.Tensor,
    model: nn.Module,
    device: str = "cuda"
) -> torch.Tensor:
    """Process video frames through the model.

    Args:
        frames: Input frames of shape (B, T, C, H, W)
        model: The neural network model
        device: Device to run computation on

    Returns:
        Processed features of shape (B, T, D)
    """
    pass
```

## Types of Contributions

### 1. Bug Fixes

- Check existing issues for similar bug reports
- Add unit tests to prevent regression
- Document the fix

### 2. New Features

- Open an issue to discuss the feature first
- Update documentation
- Add examples

### 3. Documentation

- Fix typos and grammatical errors
- Improve clarity
- Add tutorials

### 4. Performance Improvements

- Benchmark changes
- Profile code before and after
- Document improvements

## Testing

### Unit Tests

```bash
pytest tests/
```

### Integration Tests

```bash
pytest tests/integration/
```

### Code Coverage

```bash
pytest --cov=jain_mf tests/
```

## Review Process

1. Automated checks run on pull requests
2. Code review by maintainers
3. Required approvals before merge

## Release Process

Releases are tagged using semantic versioning:

- `MAJOR.MINOR.PATCH`
- Breaking changes: increment `MAJOR`
- New features: increment `MINOR`
- Bug fixes: increment `PATCH`