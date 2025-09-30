#!/bin/bash
# Lint code with ruff

echo "🔍 Linting code with ruff..."
uv run ruff check . --fix

echo "✅ Linting complete!"