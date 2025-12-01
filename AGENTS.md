# Agent Guidelines for AI Trading System

## Build/Test Commands
- **Run all tests**: `pytest tests/`
- **Single test file**: `pytest tests/test_risk_manager.py`
- **Single test**: `pytest tests/test_risk_manager.py::test_position_size_calculation -v`
- **Lint**: `ruff check .` (auto-fix: `ruff check --fix .`)
- **Format**: `black . && isort .`
- **Type check**: `mypy src/`
- **Run bot**: `PYTHONPATH=/root/ai_trader python src/main.py continuous --auto-restart`

## Code Style
- **Imports**: stdlib → third-party → local (isort profile=black, line_length=88)
- **Formatting**: Black defaults (88 chars), double quotes
- **Types**: Type hints required on all functions; use `Optional[]`, `Dict[]`, `List[]`
- **Naming**: snake_case (funcs/vars), PascalCase (classes), UPPER_CASE (constants)
- **Error Handling**: Always `except Exception as e:`, log with `logger.error("event", error=str(e))`
- **Docstrings**: Google style, required for public functions
- **Config**: Use `config.config` for settings, env vars via `python-dotenv`

## Trading System Rules
- **Risk**: Validate 2% position sizing via `risk_manager.calculate_position_size()` before trades
- **Logging**: Use `structlog` with JSON format; include trade reasoning for audit trail
- **API Calls**: Alpaca for trading/data, NewsAPI for sentiment; mock all external APIs in tests
- **Testing**: Mock external APIs with `pytest-mock`; test risk calculations extensively
