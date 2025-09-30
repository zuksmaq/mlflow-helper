#!/bin/bash
# Type check with mypy

echo "🔬 Type checking with mypy..."
uv run mypy src/

echo "✅ Type checking complete!"