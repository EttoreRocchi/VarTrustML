# Contributing to VarTrustML

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/EttoreRocchi/VarTrustML.git
   cd VarTrustML
   ```

2. Install in development mode:
   ```bash
   pip install -e '.[dev]'
   ```

3. (Optional) Install pre-commit hooks for automatic checks on commit:
   ```bash
   pre-commit install
   ```

## Code Style

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting.

Run checks manually:
```bash
pre-commit run --all-files
```

If you ran `pre-commit install`, checks run automatically on every `git commit`.

## Testing

Run the test suite:
```bash
pytest
```

## Pull Requests

1. Create a feature branch from `main`
2. Make your changes
3. Ensure tests pass and pre-commit hooks succeed
4. Submit a PR with a clear description of changes

## Reporting Issues

Please use [GitHub Issues](https://github.com/EttoreRocchi/VarTrustML/issues) to report bugs or request features.
