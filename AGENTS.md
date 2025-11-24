# Agent Guidelines for AI Trading System

## Build/Test Commands
- Python tests: `pytest tests/` (single test: `pytest tests/test_file.py::test_name`)
- Lint: `ruff check .` (auto-fix: `ruff check --fix .`)
- Format: `black . && isort .`
- Type check: `mypy src/`

## Code Style
- **Imports**: stdlib → third-party → local (use `isort` profile black)
- **Formatting**: Black (line length 88), double quotes for strings
- **Types**: Use type hints for all function signatures, return types required
- **Naming**: snake_case for functions/variables, PascalCase for classes, UPPER_CASE for constants
- **Error Handling**: Use explicit exceptions, never bare `except:`, log all errors with context
- **Async**: Prefer async/await for I/O operations (API calls, file operations)
- **Documentation**: Docstrings for all public functions (Google style), inline comments for complex logic
- **API Keys**: Never hardcode, use environment variables via `python-dotenv`

## Trading System Specifics
- **Risk Management**: Always validate 2% position sizing before execution
- **Logging**: Use structured logging (JSON) with trade reasoning for audit trail
- **API Calls**: Implement exponential backoff, respect rate limits (EODHD: 20 req/day free tier)
- **Testing**: Mock all external APIs (Alpaca, EODHD), test risk calculations extensively
- **Data**: Cache market data locally to minimize API usage, validate all input data
